import logging
import os
import log1

def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s-%(message)s', filemode='w', filename='myapp.log', level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info('Started')
    log1.do_something()
    logging.info('End')

if __name__ == '__main__':
    main()
