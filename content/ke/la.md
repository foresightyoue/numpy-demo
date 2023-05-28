# Markdown 添加 Latex 数学公式
https://www.cnblogs.com/peaceWang/p/Markdown-tian-jia-Latex-shu-xue-gong-shi.html

$ x^2, x_1^2, x^{(n)}_{22}, ^{16}O^{2-}_{32}, x^{y^{z^a}}, x^{y_z} $

$$
\sum_{i=0}^{n} x_i
$$
## 分式
$\frac{x+y}{y+z}$ 

$\displaystyle\frac{x+y}{y+z}$

$x_0+\frac{1}{x_1+\frac{1}{x_2+\frac{1}{x_3+\frac{1}{x_4}}}}$

$\newcommand{\FS}[2]{\displaystyle\frac{#1}{#2}}x0+\FS{1}{X_1+\FS{1}{X_2+\FS{1}{X_3+\FS{1}{X_4}}}}$

$\frac{1}{2},\frac{\;1\;}{\;2\;}$

## 根式
$\sqrt{a}+\sqrt{b}+\sqrt{c},\qquad \sqrt{\mathstrut a}+\sqrt{\mathstrut b}+\sqrt{\mathstrut c}$

$\sqrt{1+\sqrt[p]{1+\sqrt[q]{1+a}}}$

[n]改为[^n\!],其中^表示是上标，\!表示缩小间隔，

$\sqrt{1+\sqrt[^p\!]{1+\sqrt[^q\!]{1+a}}}$

## 求和与积分
无穷级数
$\sum_{k=1}^\infty\frac{x^n}{n!}$
可以化为积分
$\int_0^\infty e^x$
改变上下限位置的命令：\limits(强制上下限在上下侧) 和 \nolimits(强制上下限在左右侧)
$\sum\limits_{k=1}^n$ 和 $\sum\nolimits_{k=1}^n$


## 下划线、上划线等
$\overline{\overline{a^2}+\underline{ab}+\bar{a}^3}$

上花括弧命令：\overbrace{公式}{说明}
下花括弧命令：\underbrace{公式}_{说明}

$\underbrace{a+\overbrace{b+\dots+b}^{m\mbox{\tiny 个}}}_{20\mbox{\scriptsize 个}}$

## 数学重音符号

$$
\hat{a}
\check{a}
\breve{a}
\tilde{a}
\bar{a}
\vec{a}
\acute{a}
\grave{a}
\mathring{a}
\dot{a}
\ddot{a}
$$

## 堆积符号
\stacrel{上位符号}{基位符号} 基位符号大，上位符号小
{上位公式\atop 下位公式} 上下符号一样大
{上位公式\choose 下位公式} 上下符号一样大；上下符号被包括在圆弧内

$ \vec{x}\stackrel{\mathrm{def}}{=}{x_1,\dots,x_n}\\ {n+1 \choose k}={n \choose k}+{n \choose k-1}\\ \sum_{k_0,k_1,\ldots>0 \atop k_0+k_1+\cdots=n}A_{k_0}A_{k_1}\cdots $

## 定界符
$$
（）\big(\big) \Big(\Big) \bigg(\bigg) \Bigg(\Bigg)
\big(\Big) \bigg(\Bigg)
$$

自适应放大命令：\left 和\right，本命令放在左右定界符前，自动随着公式内容大小调整符号大小

## 占位宽度
两个quad空格 a \qquad b 两个m的宽度, 显示为：ab
quad空格 a \quad b 一个m的宽度，显示为ab
大空格 a\ b 1/3m宽度，显示为：a b
中等空格 a\;b 2/7m宽度，显示为：ab
小空格 a\,b 1/6m宽度, 显示为：ab
没有空格 ab, 显示为：ab
紧贴 a\!b 缩进1/6m宽度, 显示为：ab
\quad代表当前字体下接近字符‘M’的宽度（approximately the width of an "M" in the current font）

## 集合相关的运算命令
集合的大括号： \{ ...\}，显示为：{...}
集合中的|： , 显示为： ∣
属于： \in 显示为： ∈
不属于： \not\in 显示为： ∉
A包含于B： A\subset B显示为：A⊂B
A真包含于B：A \subsetneqq B 显示为：A⫋B
A包含B：A supset B 显示为：A⊃B
A真包含B：A \supsetneqq B 显示为: A⫌B
A不包含于B：A \not \subset B 显示为：A⊄B
A交B： A \cap B 显示为：A∩B
A并B：A \cup B 显示为：A∪B
A的闭包：\overline{A}显示为：A¯¯¯¯
A减去B: A\setminus B显示为：A∖B
实数集合： \mathbb{R} 显示为：R
空集：\emptyset 显示为：∅


$ y_N, y_{_N}, y_{_{\scrptstyle N} $

$ \partial f_{\mbox{\tiny 极大值}} $

# 常用经典公式
$$
\begin{array}{c}   \text{若}P \left( AB \right) =P \left( A \right) P \left( B \right) \\    \text{则}P \left( A \left| B\right. \right) =P \left({B}\right) \end{array}
$$

## AI/机器学习常用公式的LaTex代码汇总
https://zhuanlan.zhihu.com/p/142572147


$$
s_t = RNN_{dec}(y_t, s_{t-1})
$$


## AI/机器学习常用公式的LaTex代码汇总
https://blog.csdn.net/blmoistawinde/article/details/106258983

https://github.com/blmoistawinde/ml_equations_latex

### Variational Auto-Encoder(VAE)
$$
z \sim q_{\mu, \sigma}(z) = \mathcal{N}(\mu, \sigma^2)
\epsilon \sim \mathcal{N}(0,1)
z = \mu + \epsilon \cdot \sigma
$$

### GitHub项目awesome-latex-drawing新增内容（四）：绘制贝叶斯网络
https://www.zhihu.com/tardis/zm/art/25067183?source_id=1005

## mathModel
https://github.com/zhanwen/MathModel

## latex2sympy 
https://github.com/augustt198/latex2sympy

## latex常用公式
https://www.cnblogs.com/LonelyStevenL/p/17427142.html