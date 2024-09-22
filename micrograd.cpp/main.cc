#include <iostream>

#include "src/engine.h"
#include "src/nn.h"


void TestValue() {
  auto a = std::make_shared<micrograd::Value>(7);
  auto b = std::make_shared<micrograd::Value>(3);
  auto c = std::make_shared<micrograd::Value>(2);
  auto d = std::make_shared<micrograd::Value>(5);
  auto res = a->Sub(b);
  res = res->Mul(c);
  res = res->Add(d);
  res->Backward();
  res->Print();
  a->Print();
  b->Print();
  c->Print();
  d->Print();
}


void TestMLP() {
  std::vector<float> w1{1.0f, 1.0f, 1.0f};
  auto n1 = std::make_shared<micrograd::Neuron>(3, w1, 1.0f);
  std::vector<float> w2{1.0f, 2.0f, 3.0f};
  auto n2 = std::make_shared<micrograd::Neuron>(3, w2, 0.02f);
  std::vector<micrograd::NeuronPtr> ln1{n1, n2};
  auto l1 = std::make_shared<micrograd::Layer>(3, 2, ln1);

  std::vector<float> w3{3.0f, 3.0f};
  auto n3 = std::make_shared<micrograd::Neuron>(2, w3, 6.9f);
  std::vector<micrograd::NeuronPtr> ln2{n3};
  auto l2 = std::make_shared<micrograd::Layer>(2, 1, ln2);

  std::vector<micrograd::LayerPtr> ls{l1, l2};
  auto mlp = std::make_shared<micrograd::MLP>(ls);

  auto x1 = std::make_shared<micrograd::Value>(1);
  auto x2 = std::make_shared<micrograd::Value>(2);
  auto x3 = std::make_shared<micrograd::Value>(3);
  std::vector<micrograd::ValuePtr> x{x1, x2, x3};

  x = mlp->Forward(x);
  for (const auto& val : x) {
    val->Print();
  }
}


int main() {
  //  TestValue();
  TestMLP();
}
