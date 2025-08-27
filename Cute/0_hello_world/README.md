# my cute notes
## GEMM

[GEMM详细介绍](./1_GEMM/readme.md)



## print mma 流程
```bash
# nvcc 编译
nvcc mma.cu -o print_mma

# 获取tex输出
./print_mma > mma.tex

# 安装转换pdf package
apt-get update
apt install texlive-latex-base -y
apt-get install texlive-latex-extra -y
apt install imagemagick

# 编译pdf 
pdflatex mma.tex
