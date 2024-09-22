//
// Created by binhtranmcs on 17/09/2024.
//

#ifndef MICROGRAD_CPP_SRC_ENGINE_H_
#define MICROGRAD_CPP_SRC_ENGINE_H_

#include <cmath>
#include <memory>
#include <vector>


namespace micrograd {


class Value;
using ValuePtr = std::shared_ptr<Value>;


class Value : public std::enable_shared_from_this<Value> {
public:
  explicit Value(float value = 0) : value_(value) {
  }

  ValuePtr Add(std::shared_ptr<Value> other) {
    auto out = std::make_shared<Value>(value_ + other->value_);
    out->backward_mul_ = {1, 1};
    out->prev_ = {shared_from_this(), std::move(other)};
    return out;
  }

  ValuePtr Mul(ValuePtr other) {
    auto out = std::make_shared<Value>(value_ * other->value_);
    out->backward_mul_ = {other->value_, value_};
    out->prev_ = {shared_from_this(), std::move(other)};
    return out;
  }

  ValuePtr Pow(float e) {
    auto out = std::make_shared<Value>(std::pow(value_, e));
    out->backward_mul_ = {e * std::pow(value_, e - 1)};
    out->prev_ = {shared_from_this()};
    return out;
  }

  ValuePtr Neg() {
    return Mul(std::make_shared<Value>(-1));
  }

  ValuePtr Sub(const ValuePtr& other) {
    return Add(other->Neg());
  }

  ValuePtr Div(const ValuePtr& other) {
    return Mul(other->Pow(-1));
  }

  ValuePtr Relu() {
    float value = value_ > 0 ? value_ : 0;
    auto out = std::make_shared<Value>(value);
    out->backward_mul_ = {(value_ > 0) ? 1.0f : 0.0f};
    out->prev_ = {shared_from_this()};
    return out;
  }

  float Val() const {
    return value_;
  }

  void Print() const {
    std::cout << "value: " << value_ << '\n';
    std::cout << "grad: " << grad_ << '\n';
    std::cout << "-----\n";
  }

  void UpdateGrad(float grad) {
    grad_ += grad;
  }

  void BackwardFunc() {
    for (size_t i = 0; i < prev_.size(); ++i) {
      prev_[i]->UpdateGrad(grad_ * backward_mul_[i]);
    }
  }

  void Backward() {
    grad_ = 1;
    auto topo = BuildTopo();
    for (auto& val : topo) {
      val->BackwardFunc();
    }
  }

private:
  std::vector<ValuePtr> BuildTopo() { // NOLINT
    std::vector<ValuePtr> res;
    res.push_back(shared_from_this());
    for (auto& prev : prev_) {
      auto topo = prev->BuildTopo();
      res.insert(res.end(), topo.begin(), topo.end());
    }
    return res;
  }

  float value_;
  float grad_{};

  std::vector<ValuePtr> prev_;
  std::vector<float> backward_mul_;
};


} // namespace micrograd


#endif // MICROGRAD_CPP_SRC_ENGINE_H_
