#include <torch/extension.h>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}

/*"""
#include <torch/extension.h>:

这是 PyTorch 扩展的头文件，它包含了所有需要的东西，包括 pybind11 的核心功能和 PyTorch 张量（torch::Tensor）与 Python 张量之间的转换逻辑。
torch::Tensor forward(...):

这是一个 C++ 函数声明。它告诉编译器，存在一个名为 forward 的函数，它接收三个 torch::Tensor 类型的参数，并返回一个 torch::Tensor。
这个函数的具体定义（实现）在 flash.cu 文件中（代码中未显示，但这是它的作用）。
PYBIND11_MODULE(...):

这是一个宏，是 pybind11 绑定的入口点。当 Python import 这个编译好的模块时，这段代码块就会执行。
TORCH_EXTENSION_NAME: 这是一个非常方便的宏，它会自动被替换为你在 Python load() 函数中指定的 name。在这里，它会被替换为 "minimal_attn"。
m: 这是一个 pybind11::module_ 类型的变量，它代表了我们正在创建的 Python 模块本身。
m.def("forward", ...):

这是最核心的绑定语句。.def() 方法是在模块 m 上定义一个新函数。
"forward": 这是该函数在 Python 中暴露的名字。所以你可以在 Python 中调用 minimal_attn.forward()。
torch::wrap_pybind_function(forward): 这是 PyTorch 提供的魔法包装器。
里面的 forward 是指向我们 C++ 函数 ::forward 的函数指针。
torch::wrap_pybind_function 会自动处理所有棘手的细节：
将 Python 的 torch.Tensor 对象转换成 C++ 的 torch::Tensor 对象。
管理内存和数据类型。
处理设备信息（CPU/GPU）。
将 C++ 函数返回的 torch::Tensor 再转换回 Python 的 torch.Tensor 对象。
"forward" (最后的字符串): 这是该函数的文档字符串（docstring），当你在 Python 中执行 help(minimal_attn.forward) 时会显示。
"""
*/