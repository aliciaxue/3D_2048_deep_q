"""Tests for `Game` class."""

from py_2048_rl.game.game import Game
from mock import call, patch

import numpy as np

# pylint: disable=missing-docstring

@patch('numpy.random.choice')
def test_init(choice):
  choice.side_effect = [0,  # First position
                        1,  # First tile
                        1,  # Second position
                        2]  # Second tile
  game = Game()

  choice.assert_has_calls([call(27),
                           call([2, 4], p=[0.8, 0.2]),
                           call(26),
                           call([2, 4], p=[0.8, 0.2])])

  # Assert correct number of 0s, 1s and 2s
  game.print_state()
  assert (np.bincount(game.state().flatten()) == [25, 1, 1]).all()
  assert game.score() == 0


def test_available_actions():
  state = np.array([[[1, 2, 0],
                     [1, 2, 0],
                     [1, 2, 0]],

                    [[1, 2, 0],
                     [1, 2, 0],
                     [1, 2, 0]],

                    [[1, 2, 0],
                     [1, 2, 0],
                     [1, 2, 0]]]
                    )

  game = Game(state=state)
  actions = game.available_actions()

  # All actions except left is available
  assert actions == [1, 2, 3, 4, 5]


def test_available_actions_none_available():
  state = np.array([[[1, 2, 3],
                     [5, 6, 7],
                     [1, 2, 3]],

                    [[8, 9, 10],
                     [11, 12, 13],
                     [8, 9, 10]],

                    [[14, 15, 16],
                     [17, 18, 19],
                     [14, 15, 16]]]
                    )

  game = Game(state=state)
  actions = game.available_actions()

  # All actions except left is available
  assert actions == []
  assert game.game_over()


@patch('numpy.random.choice')
def test_do_action(choice):
  choice.side_effect = [0,  # First position
                        1]  # First tile
  state = np.array([[[1, 2, 3],
                     [5, 6, 7],
                     [5, 2, 7]],

                    [[1, 2, 3],
                     [5, 6, 7],
                     [5, 2, 7]],

                    [[1, 2, 3],
                     [5, 6, 7],
                     [5, 2, 7]]]
                     )

  game = Game(state=state)
  game.do_action(3)  # DOWN

  new_state = np.array([[[0, 2, 0],
                         [1, 6, 3],
                         [6, 2, 8]],

                        [[0, 2, 0],
                         [1, 6, 3],
                         [6, 2, 8]],

                        [[0, 2, 0],
                         [1, 6, 3],
                         [6, 2, 8]]]
                     )
  game.print_state()
  assert (game.state() == new_state).all()
  # Score is (2 ** 6 + 2 ** 8)*3
  assert game.score() == 960
