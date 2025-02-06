// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace tt::fabric {

enum TerminationSignal : uint32_t {
    KEEP_RUNNING = 0,

    // Wait for messages to drain
    GRACEFULLY_TERMINATE = 1,

    // Immediately terminate - don't wait for any outstanding messages to arrive or drain out
    IMMEDIATELY_TERMINATE = 2
};


// 2 bits
enum NocSendType : uint8_t {
    NOC_UNICAST_WRITE = 0,
    NOC_MULTICAST_WRITE = 1,
    NOC_UNICAST_ATOMIC_INC = 2,
    NOC_MULTICAST_ATOMIC_INC = 3
};
// How to send the payload across the cluster
// 1 bit
enum ChipSendType : uint8_t {
    CHIP_UNICAST = 0,
    CHIP_MULTICAST = 1,
};


struct UnicastRoutingCommandHeader {
    uint8_t distance_in_hops;
};
static_assert(sizeof(UnicastRoutingCommandHeader) == 1, "UnicastRoutingCommandHeader size is not 1 byte");
struct MulticastRoutingCommandHeader {
    uint8_t start_distance_in_hops: 4;
    uint8_t range_hops: 4; // 0 implies unicast
};
static_assert(sizeof(MulticastRoutingCommandHeader) == 1, "MulticastRoutingCommandHeader size is not 1 byte");
union RoutingFields {
    UnicastRoutingCommandHeader chip_unicast;
    MulticastRoutingCommandHeader chip_mcast;
};
static_assert(sizeof(RoutingFields) == sizeof(UnicastRoutingCommandHeader), "RoutingFields size is not 1 bytes");

struct NocUnicastCommandHeader {
    uint64_t noc_address;
    uint32_t size;
    // ignores header size
    inline uint32_t get_payload_only_size() const {
        return size;
    }
};
struct NocUnicastAtomicIncCommandHeader {
    NocUnicastAtomicIncCommandHeader(uint64_t noc_address, uint16_t val, uint16_t wrap)
        : noc_address(noc_address), val(val), wrap(wrap) {}

    uint64_t noc_address;
    uint16_t val;
    uint16_t wrap;
};
struct NocMulticastCommandHeader {
    uint32_t address;
    uint32_t size;
    uint8_t noc_x_start;
    uint8_t noc_y_start;
    uint8_t mcast_rect_size_x;
    uint8_t mcast_rect_size_y;

    // ignores header size
    inline uint32_t get_payload_only_size() const {
        return size;
    }
};
struct NocMulticastAtomicIncCommandHeader {
    uint32_t address;
    uint16_t val;
    uint16_t wrap;
    uint8_t noc_x_start;
    uint8_t noc_y_start;
    uint8_t size_x;
    uint8_t size_y;
};
static_assert(sizeof(NocUnicastCommandHeader) == 16, "NocUnicastCommandHeader size is not 1 byte");
static_assert(sizeof(NocMulticastCommandHeader) == 12, "NocMulticastCommandHeader size is not 1 byte");
static_assert(sizeof(NocUnicastAtomicIncCommandHeader) == 16, "NocUnicastCommandHeader size is not 1 byte");
static_assert(sizeof(NocMulticastAtomicIncCommandHeader) == 12, "NocAtomicIncCommandHeader size is not 1 byte");
union NocCommandFields{
    NocUnicastCommandHeader unicast_write;
    NocMulticastCommandHeader mcast_write;
    NocUnicastAtomicIncCommandHeader unicast_seminc;
    NocMulticastAtomicIncCommandHeader mcast_seminc;
} ;
static_assert(sizeof(NocCommandFields) <= 16, "CommandFields size is not 16 bytes");

// TODO: wrap this in a debug version that holds type info so we can assert for field/command/
struct PacketHeader {
    // TODO: trim this down noc_send_type 2 bits (4 values):
    //   -> unicast_write, mcast_write, unicast_seminc, mcast_seminc
    // For now, kept it separate so I could do reads which would be handled differently
    // but for our purposes we shouldn't need read so we should be able to omit the support
    NocSendType noc_send_type : 2;
    ChipSendType chip_send_type : 1;
    uint8_t reserved : 4;

    RoutingFields routing_fields;
    uint16_t reserved2; // can be tagged with src device for debug
    NocCommandFields command_fields; // size = 16B due to uint64_t alignment

    // Sort of hack to work-around DRAM read alignment issues that must be 32B aligned
    // To simplify worker kernel code, we for now decide to pad up the packet header
    // to 32B so the user can simplify shift into their CB chunk by sizeof(tt::fabric::PacketHeader)
    // and automatically work around the DRAM read alignment bug.
    //
    // Future changes will remove this padding and require the worker kernel to be aware of this bug
    // and pad their own CBs conditionally when reading from DRAM. It'll be up to the users to
    // manage this complexity.
    uint32_t padding0;
    uint32_t padding1;

    inline void set_chip_send_type(ChipSendType &type) { this->chip_send_type = type; }
    inline void set_noc_send_type(NocSendType &type) { this->noc_send_type = type; }
    inline void set_routing_fields(RoutingFields &fields) { this->routing_fields = fields; }
    inline void set_command_fields(NocCommandFields &fields) { this->command_fields = fields; }

    size_t get_payload_size_excluding_header() volatile const {
        switch(this->noc_send_type) {
            case NOC_UNICAST_WRITE: {
                return this->command_fields.unicast_write.size - sizeof(PacketHeader);
            } break;
            case NOC_MULTICAST_WRITE: {
                return this->command_fields.mcast_write.size - sizeof(PacketHeader);
            } break;
            case NOC_UNICAST_ATOMIC_INC:
            case NOC_MULTICAST_ATOMIC_INC:
                return 0;
            default:
            #if defined(KERNEL_BUILD) || defined(FW_BUILD)
                ASSERT(false);
            #endif
                return 0;
        };
    }
    inline size_t get_payload_size_including_header() volatile const {
        return get_payload_size_excluding_header() + sizeof(PacketHeader);
    }

    inline PacketHeader &to_chip_unicast(UnicastRoutingCommandHeader const &chip_unicast_command_header) {
        this->chip_send_type = CHIP_UNICAST;
        this->routing_fields.chip_unicast = chip_unicast_command_header;
        return *this;
    }
    inline PacketHeader &to_chip_multicast(MulticastRoutingCommandHeader const &chip_multicast_command_header) {
        this->chip_send_type = CHIP_MULTICAST;
        this->routing_fields.chip_mcast = chip_multicast_command_header;
        return *this;
    }

    inline PacketHeader &to_noc_unicast_write(NocUnicastCommandHeader const &noc_unicast_command_header) {
        this->noc_send_type = NOC_UNICAST_WRITE;
        this->command_fields.unicast_write = noc_unicast_command_header;
        return *this;
    }
    inline PacketHeader &to_noc_multicast_write(NocMulticastCommandHeader const &noc_multicast_command_header) {
        this->noc_send_type = NOC_MULTICAST_WRITE;
        this->command_fields.mcast_write = noc_multicast_command_header;
        return *this;
    }
    inline PacketHeader &to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader const &noc_unicast_atomic_inc_command_header) {
        this->noc_send_type = NOC_UNICAST_ATOMIC_INC;
        this->command_fields.unicast_seminc = noc_unicast_atomic_inc_command_header;
        return *this;
    }
    inline PacketHeader &to_noc_multicast_atomic_inc(NocMulticastAtomicIncCommandHeader const &noc_multicast_command_header) {
        #if defined(KERNEL_BUILD) || defined(FW_BUILD)
        ASSERT(false);
        while (1) {};
        #endif
        return *this;
    }

    inline volatile PacketHeader *to_chip_unicast(UnicastRoutingCommandHeader const &chip_unicast_command_header) volatile {
        this->chip_send_type = CHIP_UNICAST;
        this->routing_fields.chip_unicast.distance_in_hops = chip_unicast_command_header.distance_in_hops;
        return this;
    }
    inline volatile PacketHeader *to_chip_multicast(MulticastRoutingCommandHeader const &chip_multicast_command_header) volatile {
        this->chip_send_type = CHIP_MULTICAST;
        this->routing_fields.chip_mcast.range_hops = chip_multicast_command_header.range_hops;
        this->routing_fields.chip_mcast.start_distance_in_hops = chip_multicast_command_header.start_distance_in_hops;
        return this;
    }
    inline volatile PacketHeader *to_noc_unicast(NocUnicastCommandHeader const &noc_unicast_command_header) volatile {
        DPRINT << "TO_NOC_UNICAST\n";
        this->noc_send_type = NOC_UNICAST_WRITE;
        this->command_fields.unicast_write.noc_address = noc_unicast_command_header.noc_address;
        this->command_fields.unicast_write.size = noc_unicast_command_header.size;

        return this;
    }
    inline volatile PacketHeader *to_noc_multicast(NocMulticastCommandHeader const &noc_multicast_command_header) volatile {
        DPRINT << "TO_NOC_MULTICAST\n";
        this->noc_send_type = NOC_MULTICAST_WRITE;
        this->command_fields.mcast_write.mcast_rect_size_x = noc_multicast_command_header.mcast_rect_size_x;
        this->command_fields.mcast_write.mcast_rect_size_y = noc_multicast_command_header.mcast_rect_size_y;
        this->command_fields.mcast_write.noc_x_start = noc_multicast_command_header.noc_x_start;
        this->command_fields.mcast_write.noc_y_start = noc_multicast_command_header.noc_y_start;
        this->command_fields.mcast_write.size = noc_multicast_command_header.size;
        this->command_fields.mcast_write.address = noc_multicast_command_header.address;

        return this;
    }
    inline volatile PacketHeader *to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader const &noc_unicast_atomic_inc_command_header) volatile {
        DPRINT << "TO_NOC_UNICAST_ATOMIC_INC\n";
        this->noc_send_type = NOC_UNICAST_ATOMIC_INC;
        this->command_fields.unicast_seminc.noc_address = noc_unicast_atomic_inc_command_header.noc_address;
        this->command_fields.unicast_seminc.val = noc_unicast_atomic_inc_command_header.val;
        this->command_fields.unicast_seminc.wrap = noc_unicast_atomic_inc_command_header.wrap;

        return this;
    }
    inline volatile PacketHeader *to_noc_multicast_atomic_inc(
        NocMulticastAtomicIncCommandHeader const &noc_multicast_atomic_inc_command_header) volatile {
        DPRINT << "TO_NOC_MULTICAST_ATOMIC_INC\n";
        this->noc_send_type = NOC_MULTICAST_ATOMIC_INC;
        this->command_fields.mcast_seminc.address = noc_multicast_atomic_inc_command_header.address;
        this->command_fields.mcast_seminc.noc_x_start = noc_multicast_atomic_inc_command_header.noc_x_start;
        this->command_fields.mcast_seminc.noc_y_start = noc_multicast_atomic_inc_command_header.noc_y_start;
        this->command_fields.mcast_seminc.size_x = noc_multicast_atomic_inc_command_header.size_x;
        this->command_fields.mcast_seminc.size_y = noc_multicast_atomic_inc_command_header.size_y;
        this->command_fields.mcast_seminc.val = noc_multicast_atomic_inc_command_header.val;
        this->command_fields.mcast_seminc.wrap = noc_multicast_atomic_inc_command_header.wrap;

        return this;
    }
};


// TODO: When we remove the 32B padding requirement, reduce to 16B size check
static_assert(sizeof(PacketHeader) == 32, "sizeof(PacketHeader) is not equal to 32B");

static constexpr size_t header_size_bytes = sizeof(PacketHeader);


} // namespace tt::fabric
