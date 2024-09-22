//
// Created by binhtranmcs on 17/09/2024.
//

#ifndef MICROGRAD_CPP_SRC_ENGINE_H_
#define MICROGRAD_CPP_SRC_ENGINE_H_

#include <vector>
namespace micrograd {


class Value {
public:
  Value(int value, std::vector<Value> children)
      : value_(value)
      , prev_(std::move(children)) {

  }


private:
  int value_{};
  std::vector<Value> prev_;
};


}


#endif //MICROGRAD_CPP_SRC_ENGINE_H_
