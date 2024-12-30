# Greedy Kaczmarz Method with Oblique Projection 

**Authors:** Emi Cervantes, Jenny Tran

Randomized iterative algorithms are commonly used to solve large-scale linear systems of the form $Ax = b$. The Kaczmarz method is one such algorithm that utilizes only one row of $A$ in each iteration to perform orthogonal projections to solve linear systems. There are many ways to choose the next row. One such approach is a greedy method, also known as Motzkin’s Method. A variation of the Kaczmarz algorithm called Kaczmarz with oblique projections (KO), was proposed recently and utilizes oblique projections to improve convergence speed. To optimize computation, we propose the Greedy Kaczmarz with oblique projections (GKO) which chooses the row associated with the largest step length when allowing for oblique projections. Our theoretical results and numerical experiments show that in some settings, GKO performs better than alternative greedy oblique projection methods

## Experiment Descriptions

*   Synthetic experiment 1 (`synthetic-exp1.ipynb`): Compraing iteration and CPU of GK, GKO, and MWRKO.

*   Synthetic experiment 2 (`synthetic-exp2.ipynb`): Comparing iteration and CPU of GK, GKO, and MWRKO using linar syatems of various row sizes.

*   Synthetic experiment 3 (`synthetic-exp3.ipynb`): Comparing iteration and CPU of GK, GKO, and MWRKO using uniform system with various lower bounds.

*   Synthetic experiment 4 (`synthetic-exp4.ipynb`): Comparing iterationa nd CPU of GK, GKO, and MWRKO using corrrelated systems.

*   Synthetic experiment 5 (`synthetic-exp5.ipynb`): Comparing the angles of each system (not considered).

*   Synthetic experiment 6 (`synthetic-exp6.ipynb`): Comparing the convergence rates of GK, GKO, and MWRJO (not considered).