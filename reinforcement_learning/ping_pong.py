import time
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
from reinforcement_learning.reinforcement_learning_structures import *

# reinforcement learning parameters

EPISODES = 5000
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.9
SELECT_ACTION_STRATEGY = 'softmax'
EPSILON = 0.1
STATE_DIMENSION_SIZES = [10, 11, 10, 2, 2]  # xBall, yBall, xSchlaeger, xVel, yVel

# velocity value mapping

X_VEL_RIGHT_INDEX = 1
X_VEL_LEFT_INDEX = 0
Y_VEL_TOP_INDEX = 1
Y_VEL_BOTTOM_INDEX = 0

X_VEL_RIGHT_VALUE = 1
X_VEL_LEFT_VALUE = -1
Y_VEL_TOP_VALUE = 1
Y_VEL_BOTTOM_VALUE = -1


# actions: types of moving the bat


def action_left(state_properties):
    return state_properties[:2] + [state_properties[2] - 1] + state_properties[2:]


def action_stay(state_properties):
    return state_properties


def action_right(state_properties):
    return state_properties[:2] + [state_properties[2] + 1] + state_properties[2:]


class GameGL(object):
    config = None

    def __init__(self, config=None):
        self.config = config

    @staticmethod
    def to_c_string(string):
        """
        Is needed for the OpenGL-Library because standard strings are not allowed.
        """
        return bytes(string, "ascii")


class PingPongGame(GameGL, ReinforcementLearningDomain):
    """
    Ping pong game which trains an reinforcement learning agent to perform bat movement
    """

    windowName = "PingPong"
    pixelSize = 30
    xBall = 3
    yBall = 6
    xSchlaeger = 5
    xV = X_VEL_RIGHT_VALUE
    yV = Y_VEL_TOP_VALUE
    score = 0

    learning = False
    postive_rewards = 0
    total_rewards = 0

    def __init__(self, name, width=360, height=360):
        super().__init__()

        # init reinforcement learning structures
        self.actions = [action_left, action_stay, action_right]
        self.state_max_dimension_sizes = STATE_DIMENSION_SIZES
        self.agent = ReinforcementLearningAgent(self, EPISODES, DISCOUNT_FACTOR, LEARNING_RATE, SELECT_ACTION_STRATEGY,
                                                EPSILON)
        self.agent.init_state(self.get_state())
        self.learning = True
        self.agent.learn()
        self.learning = False

        # init game structures
        self.windowName = name
        self.width = width
        self.height = height

    @staticmethod
    def keyboard(key, x, y):
        # ESC = \x1w
        if key == b'\x1b':
            sys.exit(0)

    def action(self, action_index):

        actual_state = self.get_state()

        self.xSchlaeger = self.actions[action_index](actual_state)[2]

        # don't allow puncher to leave the pitch
        if self.xSchlaeger < 0:
            self.xSchlaeger = 0
        if self.xSchlaeger > self.state_max_dimension_sizes[2] - 1:
            self.xSchlaeger = self.state_max_dimension_sizes[2] - 1

        # change position regarding to velocity
        self.xBall += self.xV
        self.yBall += self.yV

        # change direction of ball if it's at wall
        if self.xBall > self.state_max_dimension_sizes[0] or self.xBall < 1:
            self.xV = X_VEL_RIGHT_VALUE if self.xV == X_VEL_LEFT_VALUE else X_VEL_LEFT_VALUE

        if self.yBall > self.state_max_dimension_sizes[1] or self.yBall < 1:
            self.yV = Y_VEL_TOP_VALUE if self.yV == Y_VEL_BOTTOM_VALUE else Y_VEL_BOTTOM_VALUE

        # check whether ball on bottom line
        if self.yBall == 0:
            self.agent.terminate()
            # check whether ball is at position of player
            if self.xSchlaeger == self.xBall or self.xSchlaeger == self.xBall - 1 or self.xSchlaeger == self.xBall - 2:
                # print("positive reward")
                reward = 1
                if not self.learning:
                    self.postive_rewards += 1
            else:
                # print("negative reward")
                reward = -1

            if not self.learning:
                self.total_rewards += 1
                print('{}% success rate'.format(round((self.postive_rewards / self.total_rewards) * 100)))

        else:
            # print('no reward')
            reward = 0

        # reset values ist hier bereits nötig, da sonst state mit nicht erlaubten / größeren Werten zurückgegeben wird
        if self.xBall > self.state_max_dimension_sizes[0]:
            self.xBall = self.state_max_dimension_sizes[0]
        elif self.xBall < 0:
            self.xBall = 0

        if self.yBall > self.state_max_dimension_sizes[1]:
            self.yBall = self.state_max_dimension_sizes[1]
        elif self.yBall < 0:
            self.yBall = 0

        print(self.get_state_parameter_value_indices())
        return reward, self.get_state_parameter_value_indices()

    def get_state(self):
        return [self.xBall, self.yBall, self.xSchlaeger, self.xV, self.yV]

    def get_state_parameter_value_indices(self):
        # print(self.xBall - 1, self.yBall - 1, self.xSchlaeger - 1)
        # test = [self.xBall, self.yBall, self.xSchlaeger,
        test = [self.xBall - 1, self.yBall - 1, self.xSchlaeger,
                X_VEL_RIGHT_INDEX if self.xV != X_VEL_LEFT_VALUE else X_VEL_LEFT_INDEX,
                Y_VEL_TOP_INDEX if self.yV != Y_VEL_BOTTOM_VALUE else Y_VEL_BOTTOM_INDEX]
        return test

    def display(self):
        # clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # reset position
        glLoadIdentity()
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, self.width, 0.0, self.height, 0.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # move bat
        action_index = self.agent.state_reaction(self.get_state_parameter_value_indices())
        self.action(action_index)

        # repaint
        self.draw_ball()
        self.draw_computer()

        # timeout of 100 milliseconds
        time.sleep(0.1)

        glutSwapBuffers()

    def start(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(100, 100)
        glutCreateWindow(self.to_c_string(self.windowName))
        # self.init()
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.on_resize)
        glutIdleFunc(self.display)
        glutKeyboardFunc(self.keyboard)
        glutMainLoop()

    def update_size(self):
        self.width = glutGet(GLUT_WINDOW_WIDTH)
        self.height = glutGet(GLUT_WINDOW_HEIGHT)

    def on_resize(self, width, height):
        self.width = width
        self.height = height

    def draw_ball(self, width=1, height=1, x=5, y=6, color=(0.0, 1.0, 0.0)):
        x = self.xBall
        y = self.yBall
        xPos = x * self.pixelSize
        yPos = y * self.pixelSize
        # set color
        glColor3f(color[0], color[1], color[2])
        # start drawing a rectangle
        glBegin(GL_QUADS)
        # bottom left point
        glVertex2f(xPos, yPos)
        # bottom right point
        glVertex2f(xPos + (self.pixelSize * width), yPos)
        # top right point
        glVertex2f(xPos + (self.pixelSize * width), yPos + (self.pixelSize * height))
        # top left point
        glVertex2f(xPos, yPos + (self.pixelSize * height))
        glEnd()

    def draw_computer(self, width=3, height=1, x=0, y=0, color=(1.0, 0.0, 0.0)):
        x = self.xSchlaeger
        xPos = x * self.pixelSize
        # set a bit away from bottom
        yPos = y * self.pixelSize  # + (self.pixelSize * height / 2)
        # set color
        glColor3f(color[0], color[1], color[2])
        # start drawing a rectangle
        glBegin(GL_QUADS)
        # bottom left point
        glVertex2f(xPos, yPos)
        # bottom right point
        glVertex2f(xPos + (self.pixelSize * width), yPos)
        # top right point
        glVertex2f(xPos + (self.pixelSize * width), yPos + (self.pixelSize * height / 4))
        # top left point
        glVertex2f(xPos, yPos + (self.pixelSize * height / 4))
        glEnd()


if __name__ == '__main__':
    game = PingPongGame("PingPong")
    game.start()
