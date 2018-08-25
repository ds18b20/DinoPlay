#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import pickle
from PIL import Image
from io import BytesIO
import base64
import cv2
import time
import functools


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsed_time * 1000)))
        return ret
    return wrapper


# path variables#path v
game_url = "chrome://dino"
chrome_driver_path = "/anaconda3/lib/python3.6/chromedriver"
loss_file_path = "./objects/loss_df.csv"
actions_file_path = "./objects/actions_df.csv"
q_value_file_path = "./objects/q_values.csv"
scores_file_path = "./objects/scores_df.csv"

# scripts
# create id for canvas for faster selection from DOM
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

# use id created above
# get image from canvas
getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"

# slower version to return canvas data
# getbase64Script = "canvasRunner = document.getElementsByClassName('runner-canvas')[0]; \
#  return canvasRunner.toDataURL().substring(22)"
'''
* Game class: Selenium interfacing between the python and browser
* __init__():  Launch the browser window using the attributes in chrome_options
* get_crashed() : return true if the agent as crashed on an obstacles. Gets javascript variable from game decribing the state
* get_playing(): true if game in progress, false is crashed or paused
* restart() : sends a signal to browser-javascript to restart the game
* press_up(): sends a single to press up get to the browser
* get_score(): gets current game score from javascript variables.
* pause(): pause the game
* resume(): resume a paused game if not crashed
* end(): close the browser and end the game
'''


class Game(object):
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(chrome_options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get('chrome://dino')
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(init_script)

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def press_down(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)  # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()

    @timeit
    def screen_capture(self):
        image_b64 = self._driver.execute_script(getbase64Script)
        image = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        return image


class DinoAgent(object):
    def __init__(self, game):  # takes game as input for taking actions
        self._game = game
        self.jump()  # to start the game, we need to jump once

    def is_running(self):
        return self._game.get_playing()

    def is_crashed(self):
        return self._game.get_crashed()

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()


class Game_sate:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game
        self._display = show_img()  # display the processed image on screen using openCV
        self._display.__next__()  # initialize the display coroutine


def save_obj(obj, name):
    with open('objects/' + name + '.pkl', 'wb') as f:  # dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def process_img(image):
    """
    process image data
    :param image: raw input array
    :return: processed image data
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
    image = image[:131, 65:65+131]  # [0:131, 65:65+131] crop Region of Interest(ROI)
    image = cv2.resize(image, (80, 80))
    return image


def show_img(graphs=False):
    """
    show images in a new window, using Python coroutine.
    :param graphs: select window title name
    :return: None
    """
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    game = Game()
    game.press_up()

    # init a coroutine for showing images
    coroutines = show_img()
    coroutines.__next__()
    while True:
        # read from canvas
        img = game.screen_capture()
        logging.info("raw data shape: {}".format(img.shape))

        # send new image data to coroutine
        img = process_img(img)
        logging.info("processed data shape: {}".format(img.shape))

        coroutines.send(img)
