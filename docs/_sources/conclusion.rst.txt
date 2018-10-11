Conclusion
=============================================

* Accuracy: FEM outperforms existing approaches, especially in the difficult limits of large coupling variability and small sample sizes.

* Efficiency: The FEM update is multiplicative and is not based on minimizing a cost function. As such, unlike e.g. gradient descent applied to Maximum Likelihood Inference, the update is not necessarily incremental. This is one of the reasons that FEM is significantly faster than MLE.

* Facilitation: FEM does not have any tunable parameter, while MLE with gradient descent has a learning rate parameter that has to be determined. In general, learning rates tend to become smaller as one nears the minimum, and this is another reason why MLE is slower than FEM. 

* Generality: FEM can be systematically applied to determine higher-order interactions, but this is, of course, also the case for any other approach to model determination.


