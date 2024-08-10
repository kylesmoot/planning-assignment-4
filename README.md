# **Assignment 4: Bayesian Inference**

In this assignment, you will use Bayesian inference to track the position of a single piece on the gameboard of [End of the Track](https://www.gaya-game.com/products/the-end-of-the-track), a two-player zero-sum board game, under an imperfect observation model. Please refer to Assignment 2 for the original rules of the game, which describe how the game pieces move.
**_Note_**: you don't need any code from assignment2 or assignment3 for this assignment.

**Description of the Observation Model:**

You will track the position of a **single piece** as it moves about the board. The positions of all the other pieces will remain fixed; you will know with certainty their positions at all times. However, you will be fed noisy observations of the single piece **_p_** as it moves about the board.

The noisy observation model p(s | x) of the piece **_p_**, where x is the state, and s is the observation, can be pictorially described as follows (numerical values are in percentage units), given that **_p_** is located at the purple square, and that blue squares represent occupied spaces.

- If the piece **_p_** has free space around it, then the observation model looks like this:

![image](https://github.com/user-attachments/assets/eca78b6b-16ba-4a58-af0b-f6c2a129053f)


- If the piece **_p_** is located next to an occupied space (in blue), which could be another piece or the board edge, then the probability of being on the adjacent square is zero, and the remainder gets merged with the probability of being on the purple square. Diagonally adjacent spaces do not affect the observation probabilities.
![image](https://github.com/user-attachments/assets/7f13960e-b972-4a53-b764-45fb8625e01b)
![image](https://github.com/user-attachments/assets/27801a37-b686-4d51-a322-8f512de496ad)


**Code Handout** (2 files): bayesian.py and test_bayesian.py in this repository

The `bayesian.py` file contains 5 functions which you will implement for Bayesian inference:

- **sample_observation** This implements both the observation model and sampling an observation; you will return both a sample coordinate and distribution.
- **sample_transition** This implements both the transition model and sampling a transition; you will return both a sample transition and distribution. As in the game, the transition model is deterministic.
- **initialize_belief** This implements initializing the prior, or initial belief. You will implement both uniform and dirac versions of the prior; you will return a distribution.
- **belief_update** This corresponds to updating the belief given an observation; you will return the updated belief (a distribution).
- **belief_predict** This corresponds to updating the belief given an action; you will return the updated belief (a distribution).

The `test_bayesian.py` file contains a few test cases; you should implement tests to ensure your code works as expected .


**Other Details:**

Given N pieces, you will be tracking the belief of exactly one of those pieces. The remaining N – 1 pieces will simply be used to occupy space on the board and they will not move.

A StateGenerator class is provided with bayesian.py to generate random states. You can specify a board size, as well as the number of pieces to place on the board. Given a board with R rows, C columns, and N pieces, you will track the belief of the first (index 0) of the N pieces over RC cells. The distributions you use will be represented as 2D Numpy arrays, with R rows, and C columns. However, the positions of the pieces on the board will be represented as (col, row) tuples.

While the observation model is noisy, the transition model is deterministic and conforms to the rules of the game. Actions are represented as a (dc, dr) tuple: that is, if the piece you are tracking belief over is located at (c, r), then taking action (dc, dr) moves the piece to location (c + dc, r + dr).


**Evaluating Your Code:**

We will test your implementations by checking the resulting belief after Bayesian inference on sequences of observations and actions.


**Extra Credit:**

The assignment is worth 41 points in total. There are an additional 3 extra credit points. In the test cases, each test is worth one point, and three of those tests will be extra credit.


# Submission Instructions
Please submit a zip file in Gradescope under CS388U-assignment4. Your zip file should include the following two files only:
1. `bayesian.py`
2. `test_bayesian.py`
