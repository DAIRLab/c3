#pragma once

#include "drake/systems/framework/basic_vector.h"

namespace c3 {
namespace systems {

/// TimestampedVector wraps a drake::systems::BasicVector along with a timestamp field
/// The primary purpose of this is to pass-through a message (e.g. LCM)
/// timestamp Uses a length N+1 drake::systems::BasicVector to store a vector of length N and a
/// timestamp The timestamp is stored as the final element (Nth)
template <typename T>
class TimestampedVector : public drake::systems::BasicVector<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(TimestampedVector)

  /// Constructs an empty TimestampedVector.
  TimestampedVector() = default;

  /// Initializes with the given @p size using the drake::dummy_value<T>, which
  /// is NaN when T = double.
  explicit TimestampedVector(int data_size)
      : drake::systems::BasicVector<T>(data_size + 1),
        timestep_index_(data_size),
        data_size_(data_size) {}

  /// Constructs a TimestampedVector with the specified @p data.
  explicit TimestampedVector(const drake::VectorX<T>& data)
      : TimestampedVector(data.size()) {
    drake::VectorX<T> data_timestamped = drake::VectorX<T>(data.size() + 1);
    data_timestamped.head(data.size()) = data;
    data_timestamped(data.size()) = 0;
    this->SetFromVector(data_timestamped);
  }

  /// Constructs a TimestampedVector whose elements are the elements of @p data.
  TimestampedVector(const std::initializer_list<T>& data)
      : TimestampedVector<T>(data.size()) {
    int i = 0;
    for (const T& datum : data) {
      this->SetAtIndex(i++, datum);
    }
  }

  void set_timestamp(T timestamp) {
    this->SetAtIndex(timestep_index_, timestamp);
  }

  T get_timestamp() const { return this->GetAtIndex(timestep_index_); }

  /// Copies the entire vector to a new TimestampedVector, with the same
  /// concrete implementation type.
  ///
  /// Uses the Non-Virtual Interface idiom because smart pointers do not have
  /// type covariance.
  std::unique_ptr<TimestampedVector<T>> Clone() const {
    auto clone = std::unique_ptr<TimestampedVector<T>>(DoClone());
    clone->set_value(this->get_value());
    return clone;
  }

  /// Returns the vector without the timestamp
  drake::VectorX<T> CopyVectorNoTimestamp() const {
    return this->CopyToVector().head(timestep_index_);
  }

  /// Returns a mutable vector of the data values (without timestamp)
  Eigen::Map<drake::VectorX<T>> get_mutable_data() {
    auto data = this->get_mutable_value().head(timestep_index_);
    return Eigen::Map<drake::VectorX<T>>(&data(0), data.size());
  }

  /// Returns the entire vector as a const Eigen::VectorBlock.
  const drake::VectorX<T> get_data() const {
    return this->get_value().head(timestep_index_);
  }

  // sets the data part of the vector (without timestamp)
  void SetDataVector(const Eigen::Ref<const drake::VectorX<T>>& value) {
    this->get_mutable_data() = value;
  }

  int data_size() const { return data_size_; }

 protected:
  /// Returns a new TimestampedVector containing a copy of the entire vector.
  /// Caller must take ownership, and may rely on the NVI wrapper to initialize
  /// the clone elementwise.
  ///
  /// Subclasses of TimestampedVector must override DoClone to return their
  /// covariant type.
  ///
  virtual TimestampedVector<T>* DoClone() const {
    return new TimestampedVector<T>(timestep_index_);
  }

 private:
  const int timestep_index_;
  const int data_size_;
};

}  // namespace systems
}  // namespace dairlib
