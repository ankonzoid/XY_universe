import logging
import threading
import time

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )

def wait_for_event(e):
    logging.debug('wait_for_event starting')
    event_is_set = e.wait()  # waits for event to be set before doing anything
    logging.debug('event set: %s', event_is_set)

def wait_for_event_timeout(e, t):
    while not e.isSet():
        logging.debug('wait_for_event_timeout starting')
        event_is_set = e.wait(t)   # wait for t seconds before doing anything
        logging.debug('event set: %s', event_is_set)
        if event_is_set:
            logging.debug('processing event')
        else:
            logging.debug('doing other work')

e = threading.Event()

t1 = threading.Thread(name='block', target=wait_for_event, args=(e,))
t1.start()

t2 = threading.Thread(name='non-block', target=wait_for_event_timeout, args=(e, 2))
t2.start()

logging.debug('Waiting before calling Event.set()')
time.sleep(3)
e.set()
logging.debug('Event is set')