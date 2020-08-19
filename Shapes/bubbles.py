import pygame
from pygame.locals import *
import random, math, sys
import time
import cv2
import numpy as np
import os


# Configurations
IMG_PATH = "../../Database/img/"
NPZ_PATH = "../../Database/npz/"
NUMBER_OF_DATA = 1000
FPS = 300
CIRCLE_BUBBLES = 1
SQUARE_BUBBLES = 0
BUBBLES_COLOR = (255,255,255)
SURFACE_COLOR = (0,0,0)
WIDTH, HEIGHT = 50, 50
BUBBLES_RADIUS = 5
CIRCULAR_CENTER = 25, 25

radius = 12
n_angles = 360

# movement_shape = 'circular'
movement_shape = 'square'

# Init Game
pygame.init()

# Set game configurations
surface = pygame.display.set_mode((WIDTH,HEIGHT))
clock = pygame.time.Clock()


class Circle:

    center = (CIRCULAR_CENTER)
    ang_idx = 0
    angles = np.linspace(0, 2*np.pi, 360)

    def __init__(self):
        self.color = BUBBLES_COLOR
        self.radius = BUBBLES_RADIUS
        # self.x = random.randint(self.radius+1, WIDTH-self.radius-1)
        # self.y = random.randint(self.radius+1, HEIGHT-self.radius-1)
        self.x = CIRCULAR_CENTER[0] +  int(CIRCULAR_CENTER[0]/2)
        self.y = CIRCULAR_CENTER[1]
        self.speedx = random.random()
        self.speedy = random.random()
    
    def board_collision(self):
        if self.x < self.radius or self.x > WIDTH-self.radius:
            self.speedx *= -1
        if self.y < self.radius or self.y > HEIGHT-self.radius:
            self.speedy *= -1
    
    def circle_collision(self, other_circle):
        if math.sqrt(((self.x-other_circle.x)**2)+((self.y-other_circle.y)**2)) <= (self.radius+other_circle.radius):
            self.collision(other_circle)
    
    # def collision(self):
    #     self.speedx *= -1
    #     self.speedy *= -1

    def collision(self, other_circle):
        speed = math.sqrt((self.speedx**2)+(self.speedy**2))
        diffx = -(self.x-other_circle.x)
        diffy = -(self.y-other_circle.y)
        if diffx > 0:
            if diffy > 0:
                angle = math.degrees(math.atan(diffy/diffx))
                speedx = -speed*math.cos(math.radians(angle))
                speedy = -speed*math.sin(math.radians(angle))
            elif diffy < 0:
                angle = math.degrees(math.atan(diffy/diffx))
                speedx = -speed*math.cos(math.radians(angle))
                speedy = -speed*math.sin(math.radians(angle))
        elif diffx < 0:
            if diffy > 0:
                angle = 180 + math.degrees(math.atan(diffy/diffx))
                speedx = -speed*math.cos(math.radians(angle))
                speedy = -speed*math.sin(math.radians(angle))
            elif diffy < 0:
                angle = -180 + math.degrees(math.atan(diffy/diffx))
                speedx = -speed*math.cos(math.radians(angle))
                speedy = -speed*math.sin(math.radians(angle))
        elif diffx == 0:
            if diffy > 0:
                angle = -90
            else:
                angle = 90
            speedx = speed*math.cos(math.radians(angle))
            speedy = speed*math.sin(math.radians(angle))
        elif diffy == 0:
            if diffx < 0:
                angle = 0
            else:
                angle = 180
            speedx = speed*math.cos(math.radians(angle))
            speedy = speed*math.sin(math.radians(angle))
        self.speedx = speedx
        self.speedy = speedy
    
    def move(self):
        # self.x += self.speedx
        # self.y += self.speedy

        if movement_shape == 'circular':
            if self.ang_idx == n_angles:
                self.ang_idx = 0

            x_center = radius * math.cos(self.angles[self.ang_idx])
            y_center = radius * math.sin(self.angles[self.ang_idx])

            self.ang_idx = self.ang_idx + 1

            self.x = CIRCULAR_CENTER[0] + x_center
            self.y = CIRCULAR_CENTER[1] + y_center
        
        elif movement_shape == 'square':

            if self.x == (CIRCULAR_CENTER[0] + int(CIRCULAR_CENTER[0]/2)) and (self.y < CIRCULAR_CENTER[0] + int(CIRCULAR_CENTER[0]/2)):
                self.y = self.y + 1
            elif self.x > (CIRCULAR_CENTER[0] - int(CIRCULAR_CENTER[0]/2)) and self.y == (CIRCULAR_CENTER[0] + int(CIRCULAR_CENTER[0]/2)):
                self.x = self.x - 1
            elif self.x == (CIRCULAR_CENTER[0] - int(CIRCULAR_CENTER[0]/2)) and self.y > (CIRCULAR_CENTER[0] - int(CIRCULAR_CENTER[0]/2)):
                self.y = self.y - 1
            elif self.x < (CIRCULAR_CENTER[0] + int(CIRCULAR_CENTER[0]/2)) and self.y == (CIRCULAR_CENTER[0] - int(CIRCULAR_CENTER[0]/2)):
                self.x = self.x + 1
            
            print("x: ", self.x)
            print("y: ", self.y)
    
    def show(self):
        pygame.draw.circle(surface, BUBBLES_COLOR, (int(self.x),int(HEIGHT-self.y)), self.radius)


class Square:
    def __init__(self):
        self.color = BUBBLES_COLOR
        self.radius = BUBBLES_RADIUS # Describes the circumference that cover the square
        self.x = random.randint(self.radius+1, WIDTH-self.radius-1)
        self.y = random.randint(self.radius+1, HEIGHT-self.radius-1)
        self.z = math.sqrt(2*(self.radius**2))
        self.w = math.sqrt(2*(self.radius**2))
        self.speedx = random.random()
        self.speedy = random.random()

        # self.radius = int((math.sqrt(2*(self.side**2))) / 2) # Describes the circumference that cover the square
    
    def board_collision(self):
        if self.x < self.radius or self.x > WIDTH-self.radius:
            self.speedx *= -1
        if self.y < self.radius or self.y > HEIGHT-self.radius:
            self.speedy *= -1
    
    def square_collision(self, other_square):
        if math.sqrt(((self.x-other_square.x)**2)+((self.y-other_square.y)**2)) <= (self.radius+other_square.radius):
            self.collision(other_square)
    
    # def collision(self):
    #     self.speedx *= -1
    #     self.speedy *= -1
    
    def collision(self, other_square):
        speed = math.sqrt((self.speedx**2)+(self.speedy**2))
        diffx = -(self.x-other_square.x)
        diffy = -(self.y-other_square.y)
        if diffx > 0:
            if diffy > 0:
                angle = math.degrees(math.atan(diffy/diffx))
                speedx = -speed*math.cos(math.radians(angle))
                speedy = -speed*math.sin(math.radians(angle))
            elif diffy < 0:
                angle = math.degrees(math.atan(diffy/diffx))
                speedx = -speed*math.cos(math.radians(angle))
                speedy = -speed*math.sin(math.radians(angle))
        elif diffx < 0:
            if diffy > 0:
                angle = 180 + math.degrees(math.atan(diffy/diffx))
                speedx = -speed*math.cos(math.radians(angle))
                speedy = -speed*math.sin(math.radians(angle))
            elif diffy < 0:
                angle = -180 + math.degrees(math.atan(diffy/diffx))
                speedx = -speed*math.cos(math.radians(angle))
                speedy = -speed*math.sin(math.radians(angle))
        elif diffx == 0:
            if diffy > 0:
                angle = -90
            else:
                angle = 90
            speedx = speed*math.cos(math.radians(angle))
            speedy = speed*math.sin(math.radians(angle))
        elif diffy == 0:
            if diffx < 0:
                angle = 0
            else:
                angle = 180
            speedx = speed*math.cos(math.radians(angle))
            speedy = speed*math.sin(math.radians(angle))
        self.speedx = speedx
        self.speedy = speedy
    
    def move(self):
        self.x += self.speedx
        self.y += self.speedy

    def show(self):
        pygame.draw.rect(surface, BUBBLES_COLOR,(int(self.x),int(self.y),int(self.z),int(self.w)))


class Draw:
    circles = []
    squares = []
    tensor = []
    screen = surface

    def __init__(self):
        for x in range(CIRCLE_BUBBLES):
            self.circles.append(Circle())
        for x in range(SQUARE_BUBBLES):
            self.squares.append(Square())
    
    def move(self):
        for circle in self.circles:
            circle.board_collision()

            for other_circle in self.circles:
                if circle != other_circle:
                    circle.circle_collision(other_circle)
            
            if self.squares is not None:
                for square in self.squares:
                    circle.circle_collision(square)
            
            circle.move()
            circle.show()
        
        for square in self.squares:
            square.board_collision()

            for other_square in self.squares:
                if square != other_square:
                    square.square_collision(other_square)
            
            if self.circles is not None:
                for circle in self.circles:
                    square.square_collision(circle)
            
            square.move()
            square.show()

    # def move_just_square(self):
    #     for square in self.squares:
    #         square.board_collision()

    #         for other_square in self.squares:
    #             if square != other_square:
    #                 square.square_collision(other_square)
            
    #         square.move()
    #         square.show()

    def show(self):
        surface.fill(SURFACE_COLOR)
        self.move()
        pygame.display.flip()
        clock.tick(FPS)
    
    def save(self, n):
        file = IMG_PATH + str(n) + '.png'
        pygame.image.save(surface, file)

    def img2npz(self):
        # from utils import show_array_as_img
        for f in os.listdir(IMG_PATH):
            if f.find(".png") != -1:
                img = self.img_processing("{}/{}".format(IMG_PATH, f))
                # show_array_as_img(img, 'gray')
                self.tensor.append(img)

        np.savez_compressed(NPZ_PATH + "bubbles.npz", self.tensor)
    
    def img_processing(self, img_path):
        img = cv2.imread(img_path, 0) # Convert to grayscale
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = img.astype('float32')
        img /= 255
        return img

    def close(self):
        keystate = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == QUIT or keystate[K_ESCAPE]:
                pygame.quit(); sys.exit()


def run(draw):
    for i in range(0, NUMBER_OF_DATA):
    # while True:
        draw.close()
        draw.show()
        draw.save(i+100)
    draw.img2npz()
    pygame.quit(); sys.exit()


draw = Draw()
if __name__ == '__main__': run(draw)
