import numpy as np


class User():
    '''
    Types of Questions (Rows):
    0 - Factual 
    1 - Precedure
    2 - Causal

    Types of Responses (Columns):
    0 - Informative
    1 - Real World Example
    2 - Goal or Outcome
    3 - Causal Principle 
    '''

    def __init__(self):
        # Necessary attributes
        self.weights = np.array([
            [0.85, 0.15, 0, 0],
            [0.53, 0.33, 0.12, 0.02],
            [0.69, 0.103, 0.103, 0.103]])

        self._learning_rate = 0.1  # In future, can make adaptive

        # Extra data
        self._num_questions = 0

    def incoming_question(self, question_array):
        # question_array is 1x3 weights for Fact, Prec, Caus
        # 1x4 weights for Info, RW, Goal, Caus
        # Matrix Multiplication weighs response matrix weights by appropriate question type
        self._answer_array = question_array@self.weights
        return self._answer_array

    def update_weights(self, feedback: int):  # Feedback is 1 or 0
        # For future calculations
        if feedback == 0:
            feedback = -1

        # Changes dimensionality of original arrays for multiplication
        # (4 -> 4x1)
        response_transformed = self._answer_array[:, np.newaxis].T
        # (3 -> 1x3)
        question_transformed = self._question_array[:, np.newaxis]

        # Create a matrix weighted by question type and response type
        self._response_matrix = question_transformed @ response_transformed  # 3x4

        # Changes direct based on user feedback
        self._response_matrix_transformed = self._response_matrix*self._user_response

        # Fraction added to original weights
        self.weights = self.weights + self._response_matrix_transformed*self.learning_rate

        # Makes rows sum to 1
        self._clean_weights()

    def _clean_weights(self):
        # 1x3 array of row minimums, 0 if minimum is positive
        row_minimums = np.min(self.weights, axis=1).clip(max=0)[:, np.newaxis]

        # Makes all weights positive, adds buffer so no weights are zero
        self.weights -= row_minimums-0.001

        # 1x3 array of row sums
        weights_row_sums = np.sum(self.weights, axis=1)[:, np.newaxis]

        # Divides each row by it's respective sum
        proportioned_weights = self.weights / weights_row_sums

        # Rounds weights to 3 decimal points
        self.weights = np.round(proportioned_weights, 3)

