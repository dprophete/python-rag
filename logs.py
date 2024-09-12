from termcolor import colored
from datetime import datetime

debug = False


def log(msg):
    print(colored(ppNow(), "yellow"), msg)


def logd(msg):
    global debug
    if debug:
        print(colored(ppNow(), "yellow"), msg)


def logError(msg):
    print(colored(ppNow(), "red"), msg)


def ppNow():
    return "[" + datetime.now().strftime("%H:%M:%S %p") + "]"
