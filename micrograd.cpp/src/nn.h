//
// Created by binhtranmcs on 17/09/2024.
//

#ifndef MICROGRAD_CPP_SRC_NN_H_
#define MICROGRAD_CPP_SRC_NN_H_

#include <cassert>
#include <vector>


namespace micrograd {


class Module;


class Neuron {
public:
  Neuron(int num_input, const std::vector<float>& w, float b) : nin_(num_input) {
    assert(nin_ == w.size());
    b_ = std::make_shared<Value>(b);
    for (int i = 0; i < nin_; ++i) {
      w_.push_back(std::make_shared<Value>(w[i]));
    }
  }

  [[nodiscard]] ValuePtr Forward(const std::vector<ValuePtr>& x) const {
    ValuePtr res = std::make_shared<Value>(0);
    res = res->Add(b_);
    for (int i = 0; i < nin_; ++i) {
      res = res->Add(x[i]->Mul(w_[i]));
    }
    return res;
  }

  [[nodiscard]] int Nin() const {
    return nin_;
  }

private:
  int nin_;
  std::vector<ValuePtr> w_;
  ValuePtr b_;
};


using NeuronPtr = std::shared_ptr<Neuron>;


class Layer {
public:
  Layer(int num_input, int num_output, const std::vector<std::vector<float>>& w,
      const std::vector<float>& b)
      : nin_(num_input)
      , nout_(num_output) {
    assert(nout_ == w.size());
    for (int i = 0; i < nout_; ++i) {
      assert(nin_ == w[i].size());
      neurons_.push_back(std::make_shared<Neuron>(nin_, w[i], b[i]));
    }
  }

  Layer(int num_input, int num_output, std::vector<NeuronPtr> neurons)
      : nin_(num_input)
      , nout_(num_output)
      , neurons_(std::move(neurons)) {
    for (const auto& neuron : neurons_) {
      assert(neuron->Nin() == nin_);
    }
    assert(nout_ == neurons_.size());
  }

  [[nodiscard]] std::vector<ValuePtr> Forward(const std::vector<ValuePtr>& x) const {
    std::vector<ValuePtr> res(nout_);
    for (int i = 0; i < nout_; ++i) {
      res[i] = neurons_[i]->Forward(x);
    }
    return res;
  }

  [[nodiscard]] int Nin() const {
    return nin_;
  }

  [[nodiscard]] int Nout() const {
    return nout_;
  }

private:
  int nin_, nout_;
  std::vector<NeuronPtr> neurons_;
};


using LayerPtr = std::shared_ptr<Layer>;


class MLP {
public:
  explicit MLP(std::vector<LayerPtr> layers) : layers_(std::move(layers)) {
    nlayer_ = static_cast<int>(layers_.size());
    nin_ = layers_[0]->Nin();
    nout_ = layers_.back()->Nout();
    for (int i = 1; i < nlayer_; ++i) {
      assert(layers_[i - 1]->Nout() == layers_[i]->Nin());
    }
  }

  [[nodiscard]] std::vector<ValuePtr> Forward(std::vector<ValuePtr> x) const {
    for (const auto& layer : layers_) {
      x = layer->Forward(x);
    }
    return x;
  }

private:
  int nin_, nout_, nlayer_{};
  std::vector<LayerPtr> layers_;
};


using MLPPtr = std::shared_ptr<MLP>;


} // namespace micrograd


#endif // MICROGRAD_CPP_SRC_NN_H_
