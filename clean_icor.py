import os
import shutil
import argparse


def move_match(file, target='D:\\deleted'):
    shutil.move(file, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--completed', help='completed files directory')
    parser.add_argument('-t', '--todo', help='directory of files yet to be processed')
    # https://stackoverflow.com/a/37411991/2238624
    parser.add_argument('-m', '--move', default=False, action='store_true', help='when specified: whether or not move the files to another folder')
    args = parser.parse_args()

    raw_todo = os.listdir(args.todo)
    raw_done = os.listdir(args.completed)

    todo = [f.split('.')[0] for f in raw_todo]
    done = [f.split('_processed.hdr')[0] for f in raw_done]

    set_todo = set(todo)
    set_done = set(done)

    intersection_files = set_done.intersection(set_todo)

    intersection_filepath = [os.path.join(args.todo, x+'.SEN3') for x in intersection_files]

    if len(intersection_files) > 0:
        print('Completed files present in both directories:')
        for f in intersection_filepath:
            print(f)
            if args.move:
                print('Moving file.')
                move_match(f)

    else:
        print('No intersecting files found.')

