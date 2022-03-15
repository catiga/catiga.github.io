---
title: 取模与取余的区别
top: false
cover: false
toc: true
mathjax: true
date: 2022-03-15 12:15:32
password:
summary:
tags:
categories:
- 算法
- 数学

---

## 模数和余数的计算方法
计 $ { (x, y) | (x, y) \in 实数 } $， $ m = \frac{x}{y} $，m计为除数，则有取模和取余的计算方法为
$$
v = x - m*y
$$

区别在于除数m的计数方法：
- 取余：除数m向0值靠近取证
- 取模：除数m向无穷小靠近取证

##### 很容易看出，取模和取余在除数m为正时（即 x，y 为同符号数时）结果相同，反之则不同

