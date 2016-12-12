import pygame
import random

#define varaibles for game
FPS = 60

#size of windonw
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

#size of our paddle
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60

#size our ball
BALL_WIDTH = 10
BALL_HEIGHT = 10

#Speed of our paddle or ball
PADDLE_SPEED = 2
BALL_X_SPEED = 3
BALL_Y_SPEED = 2

# rgb colors paddle and ball 
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

#initialize our screen 
screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])

def drawBall(ballXpos, ballYpos):
	ball = pygame.rect(ballXpos, ballYpos, BALL_WIDTH, BALL_HEIGHT)
	pygame.draw.rect(screen, WHITE, ball)

def drawPaddle1(paddle1YPos):
	paddle1 = pygame.rect(PADDLE_BUFFER, paddle1YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
	pygame.draw.rect(screen, WHITE, paddle1)

def drawPaddle2(paddle2YPos):
	paddle2 = pygame.rect(WINDOW_WIDTH-PADDLLE_BUFFER, paddle2YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
	pygame.draw.rect(screen, WHITE, paddle2)


def updateBall(paddle1YPos, paddle2YPos, ballXpos, ballYpos, ballXDirection, ballYDirection):

	# update x and y position
	ballXPos = ballXPos + ballXDirection * BALL_X_SPEED
	ballYpos = ballYpos + ballYDirection * BALL_Y_SPEED
	score = 0

	#check for a aollision, if the ball
	#hits the left side 
	#the switch the direction
	if (ballXPos <= PADDLE_BUFFER + PADDLE_WIDTH 
		and ballYpos + BALL_HEIGHT >= paddle1YPos 
		and ballYpos- BALL_HEIGHT<= paddle1YPos+PADDLE_HEIGHT):

		ballXDirection = 1

	elif (ballXpos<=0):

		ballXDirection = 1
		score = -1
		return [ score, paddle1YPos, paddle2YPos, ballXpos, ballYpos, ballXDirection, ballYDirection]

	if (ballXpos>= WINDOW_WIDTH- PADDLE_BUFFER 
		and ballYpos+ BALL_HEIGHT>= paddle2YPos 
		and ballYpos- BALL_HEIGHT<= paddle2YPos+ PADDLE_HEIGHT):

		ballXDirection = -1

	elif( ballXpos>= WINDOW_WIDTH- BALL_WIDTH):
		
		ballXDirection = -1
		score  = 1
		return [ score, paddle1YPos, paddle2YPos, ballXpos, ballYpos, ballXDirection, ballYDirection]

	if(ballYpos <=0 ):
		ballYpos = 0
		ballYDirection = 1

	elif( ballYpos >= WINDOW_HEIGHT- BALL_HEIGHT):
		ballYpos = WINDOW_HEIGHT - BALL_HEIGHT
		ballYDirection = -1

	return [ score, paddle1YPos, paddle2YPos, ballXpos, ballYpos, ballXDirection, ballYDirection]


def updatePaddle1(action, paddle1YPos):
	# if move up
	if (action[1] == 1):
		paddle1YPos = paddle1YPos - PADDLE_SPEED

	# if move down
	if (action[2] == 1):
		paddle1YPos = paddle1YPos + PADDLE_SPEED

	# dont let if move the screen
	if (paddle1YPos < 0):
		paddle1YPos = 0

	if (paddle1YPos> WINDOW_HEIGHT - PADDLE_HEIGHT):
		paddle1YPos = WINDOW_HEIGHT - PADDLE_HEIGHT

	return paddle1YPos



def updatePaddle2(action, ballYpos):
	# if move up
	if (action[1] == 1):
		paddle2YPos = paddle2YPos - PADDLE_SPEED

	# if move down
	if (action[2] == 1):
		paddle2YPos = paddle2YPos + PADDLE_SPEED

	# dont let if move the screen
	if (paddle1YPos < 0):
		paddle1YPos = 0

	if (paddle2YPos> WINDOW_HEIGHT - PADDLE_HEIGHT):
		paddle2YPos = WINDOW_HEIGHT - PADDLE_HEIGHT

	return paddle2Pos


class PongGame():

	def __init__(self):
		# random number for initial direction of ball
		num = random.randInt()
		#keep score
		self.tally = 0
		#initialize positions of paddle
		self.paddle1YPos = WINDOW_HEIGHT/2 - PADDLE_HEIGHT/2
		self.paddle2YPos = WINDOW_HEIGHT/2 - PADDLE_HEIGHT/2
		#ball direction defination
		self.ballXDirection = 1
		self.ballYDirection = 1
		#starting point
		self.ballXpos = WINDOW_HEIGHT/2 - BALL_WIDTH/2
		
	def getPresentFrame(self):
		
		# for each frame, call the event queue
		pygame.event.pump()
		#make background black
		screen.fill(BLACK)
		#draw our paddle
		drawPaddle1(self.paddle1YPos)
		drawPaddle2(self.paddle2YPos)
		#draw ball
		drawBall(self.ballXpos, self.ballYpos)
		#get pixels
		image_data = pygame.surfarray.array3d(pygame.display.get_surface())
		#update the windows
		pygame.display.flip()
		#return the screen data
		return image_data


	def getNextFrame(self, action):
		pygame.event.pump()
		screen.fill(BLACK)

		self.paddle1YPos = updatePaddle1(action, self.paddle1YPos)
		drawPaddle1(self.paddle1YPos)
		self.paddle2YPos = updatePaddle2(self.paddle2YPos, self.ballYpos)
		drawBall(self.ballXpos, self.ballYpos)

		image_data = pygame.surfarray.array3d(pygame.display.get_surface())
		pygame.display.flip()
		self.tally = self.tally + score

		return [score, image_data]


