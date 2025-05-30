// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include <assert.hpp>
#include <cstdint>
#include <mesh_coord.hpp>
#include <mesh_config.hpp>
#include <reflection.hpp>
#include <span.hpp>

namespace tt::tt_metal::distributed {
namespace {

// Returns a zero coordinate of dimensionality `dims`.
MeshCoordinate zero_coordinate(size_t dims) { return MeshCoordinate(tt::stl::SmallVector<uint32_t>(dims, 0)); }

// Returns the last valid coordinate for the provided `shape`.
MeshCoordinate shape_back(const SimpleMeshShape& shape) {
    tt::stl::SmallVector<uint32_t> coords;
    for (int i = 0; i < shape.dims(); i++) {
        coords.push_back(shape[i] - 1);
    }
    return MeshCoordinate(coords);
}

}  // namespace

SimpleMeshShape::SimpleMeshShape(uint32_t x) : ShapeBase({x}) { compute_strides(); }
SimpleMeshShape::SimpleMeshShape(uint32_t x, uint32_t y) : ShapeBase({x, y}) { compute_strides(); }
SimpleMeshShape::SimpleMeshShape(uint32_t x, uint32_t y, uint32_t z) : ShapeBase({x, y, z}) { compute_strides(); }

SimpleMeshShape::SimpleMeshShape(const MeshShape& legacy_shape) :
    SimpleMeshShape(legacy_shape.num_rows, legacy_shape.num_cols) {}

void SimpleMeshShape::compute_strides() {
    size_t stride = 1;
    strides_.resize(dims());
    for (int dim = dims() - 1; dim >= 0; --dim) {
        strides_[dim] = stride;
        stride *= (*this)[dim];
    }
}

size_t SimpleMeshShape::get_stride(size_t dim) const { return strides_[dim]; }

size_t SimpleMeshShape::dims() const { return size(); }
size_t SimpleMeshShape::mesh_size() const {
    return empty() ? 0 : std::accumulate(value_.begin(), value_.end(), 1, std::multiplies<size_t>());
}

bool operator==(const SimpleMeshShape& lhs, const SimpleMeshShape& rhs) = default;
bool operator!=(const SimpleMeshShape& lhs, const SimpleMeshShape& rhs) = default;

std::ostream& operator<<(std::ostream& os, const SimpleMeshShape& shape) {
    os << "SimpleMeshShape([";
    for (size_t i = 0; i < shape.dims(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape[i];
    }
    os << "])";
    return os;
}

MeshCoordinate::MeshCoordinate(uint32_t coord) : value_({coord}) {}
MeshCoordinate::MeshCoordinate(uint32_t x, uint32_t y) : value_({x, y}) {}
MeshCoordinate::MeshCoordinate(uint32_t x, uint32_t y, uint32_t z) : value_({x, y, z}) {}

MeshCoordinate::MeshCoordinate(tt::stl::Span<const uint32_t> coords) : value_(coords.begin(), coords.end()) {}

size_t MeshCoordinate::dims() const { return value_.size(); }
tt::stl::Span<const uint32_t> MeshCoordinate::coords() const { return value_; }
uint32_t MeshCoordinate::operator[](size_t dim) const { return value_[dim]; }

bool operator==(const MeshCoordinate& lhs, const MeshCoordinate& rhs) {
    return lhs.dims() == rhs.dims() && std::equal(lhs.coords().begin(), lhs.coords().end(), rhs.coords().begin());
}
bool operator!=(const MeshCoordinate& lhs, const MeshCoordinate& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream& os, const MeshCoordinate& coord) {
    os << "MeshCoordinate(" << coord.dims() << ", [";
    for (size_t dim : coord.coords()) {
        os << dim << ", ";
    }
    os << "])";
    return os;
}

MeshCoordinateRange::MeshCoordinateRange(const MeshCoordinate& start, const MeshCoordinate& end) :
    start_(start), end_(end) {
    TT_FATAL(
        start.dims() == end.dims(),
        "Start and end dimensions of a coordinate range do not match: {} != {}",
        start.dims(),
        end.dims());
    for (size_t i = 0; i < start.dims(); ++i) {
        TT_FATAL(start[i] <= end[i], "Start coordinate is greater than end coordinate: {} > {}", start, end);
    }
}

MeshCoordinateRange::MeshCoordinateRange(const SimpleMeshShape& shape) :
    MeshCoordinateRange(zero_coordinate(shape.dims()), shape_back(shape)) {}

const MeshCoordinate& MeshCoordinateRange::start_coord() const { return start_; }
const MeshCoordinate& MeshCoordinateRange::end_coord() const { return end_; }

MeshCoordinateRange::Iterator::Iterator(
    const MeshCoordinateRange* range, const MeshCoordinate& current, size_t linear_index) :
    range_(range), current_coord_(current), linear_index_(linear_index) {}

MeshCoordinateRange::Iterator& MeshCoordinateRange::Iterator::operator++() {
    ++linear_index_;

    tt::stl::SmallVector<uint32_t> new_coords(current_coord_.coords().begin(), current_coord_.coords().end());
    for (int i = new_coords.size() - 1; i >= 0; --i) {
        auto& dimension_value = new_coords[i];
        if (++dimension_value > range_->end_coord()[i]) {
            dimension_value = range_->start_coord()[i];
        } else {
            break;
        }
    }
    current_coord_ = MeshCoordinate(new_coords);
    return *this;
}
const MeshCoordinate& MeshCoordinateRange::Iterator::operator*() const { return current_coord_; }
bool MeshCoordinateRange::Iterator::operator==(const Iterator& other) const {
    return range_ == other.range_ && linear_index_ == other.linear_index_;
}
bool MeshCoordinateRange::Iterator::operator!=(const Iterator& other) const { return !(*this == other); }

MeshCoordinateRange::Iterator MeshCoordinateRange::begin() const { return Iterator(this, start_, /*linear_index=*/0); }
MeshCoordinateRange::Iterator MeshCoordinateRange::end() const {
    size_t range_size = 1;
    for (size_t i = 0; i < start_.dims(); ++i) {
        range_size *= end_[i] - start_[i] + 1;
    }
    // Set `start_` coordinate but `range_size` linear index as the wrap around condition.
    return Iterator(this, start_, range_size);
}

size_t to_linear_index(const SimpleMeshShape& shape, const MeshCoordinate& coord) {
    TT_FATAL(
        shape.dims() == coord.dims(),
        "Shape and coordinate dimensions do not match: {} != {}",
        shape.dims(),
        coord.dims());

    size_t linear_index = 0;
    for (size_t dim = 0; dim < coord.dims(); ++dim) {
        TT_FATAL(coord[dim] < shape[dim], "Coordinate {} is out of bounds for shape {}", coord, shape);
        linear_index += coord[dim] * shape.get_stride(dim);
    }
    return linear_index;
}

}  // namespace tt::tt_metal::distributed
