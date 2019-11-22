import time
from OpenGL.GLUT import *
from OpenGL.GL import *
from reinforcement_learning.reinforcement_learning_structures import *

# reinforcement learning parameters
EPISODES = 5000
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.9
SELECT_ACTION_STRATEGY = E_GREEDY  # SOFTMAX, GREEDY, E_GREEDY
EPSILON = 0.1

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


class BasicGame(GameGL, ReinforcementLearningDomain):

    windowName = "PingPong"
    pixelSize = 30
    xBall = 5
    yBall = 6
    xSchlaeger = 5
    xV = 1
    yV = 1

    skip = False
    learning = False
    positive_rewards = 0
    total_rewards = 0

    def __init__(self, name, width=360, height=360):
        super().__init__()
        self.windowName = name
        self.width = width
        self.height = height

        # init reinforcement learning structures
        self.actions = [action_left, action_stay, action_right]
        self.state_dimension_values = [
            # higher bound value is not included
            np.arange(1, 11),   # x ball
            np.arange(1, 12),   # y ball
            np.arange(0, 10),    # x bat
            np.array([-1, 1]),  # x velocity
            np.array([-1, 1])   # y velocity
        ]

        self.agent = ReinforcementLearningAgent(self, EPISODES, DISCOUNT_FACTOR, LEARNING_RATE, SELECT_ACTION_STRATEGY,
                                                EPSILON)

        self.agent.init_state(self.get_state())
        self.learning = True
        self.agent.learn()
        self.learning = False

    def action(self, action_index):

        self.xSchlaeger = self.actions[action_index](self.get_state())[2]
        # don't allow puncher to leave the pitch
        if self.xSchlaeger < self.state_dimension_values[2][0]:
            self.xSchlaeger = self.state_dimension_values[2][0]
        if self.xSchlaeger > self.state_dimension_values[2][-1]:
            self.xSchlaeger = self.state_dimension_values[2][-1]

        self.xBall += self.xV
        self.yBall += self.yV

        # change direction of ball if it's at wall
        if self.xBall > self.state_dimension_values[0][-1] or self.xBall < self.state_dimension_values[0][0]:
            self.xV = -self.xV
        if self.yBall > self.state_dimension_values[1][-1] or self.yBall < self.state_dimension_values[1][0]:
            self.yV = -self.yV

        # check whether ball on bottom line
        if self.yBall == 0:
            self.agent.state_terminated = True
            # check whther ball is at position of player
            if self.xSchlaeger == self.xBall or self.xSchlaeger == self.xBall - 1 or self.xSchlaeger == self.xBall - 2:
                # print("positive reward")
                reward = 1
                if not self.learning:
                    self.positive_rewards += 1
            else:
                reward = -1

            if not self.learning:
                self.total_rewards += 1
                print('{}% success rate'.format(round((self.positive_rewards / self.total_rewards) * 100)))
        else:
            reward = 0

        # nötig, verfälscht aber ergebnis
        if self.xBall > self.state_dimension_values[0][-1]:
            self.xBall = self.state_dimension_values[0][-1]
        elif self.xBall < self.state_dimension_values[0][0]:
            self.xBall = self.state_dimension_values[0][0]

        if self.yBall > self.state_dimension_values[1][-1]:
            self.yBall = self.state_dimension_values[1][-1]
        elif self.yBall < self.state_dimension_values[1][0]:
            self.yBall = self.state_dimension_values[1][0]

        return reward, self.get_state()

    def get_state(self):
        state = [self.xBall, self.yBall, self.xSchlaeger, self.xV, self.yV]
        x_temp = self.xBall
        y_temp = self.yBall

        if self.xBall > self.state_dimension_values[0][-1] or self.xBall < self.state_dimension_values[0][0]:
            x_temp += self.xV
        if self.yBall > self.state_dimension_values[1][-1] or self.yBall < self.state_dimension_values[1][0]:
            y_temp += self.yV
        return state

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

        action_index = self.agent.state_reaction(self.get_state())
        self.action(action_index)

        # repaint
        self.draw_ball()
        self.draw_computer()

        # timeout of 100 milliseconds
        time.sleep(0.01)

        glutSwapBuffers()

    @staticmethod
    def keyboard(key, x, y):
        # ESC = \x1w
        if key == b'\x1b':
            sys.exit(0)

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
    game = BasicGame("PingPong")
    game.start()