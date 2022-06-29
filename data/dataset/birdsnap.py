# Download the images in the Birdsnap dataset.
import argparse, datetime, hashlib, os, shutil
import requests

class DownloadResult(object):
    NEW_OK = 0
    ALREADY_OK = 1
    DOWNLOAD_FAILED = 2
    SAVE_FAILED = 3
    MD5_FAILED = 4
    MYSTERY_FAILED = 5
    values = (NEW_OK, ALREADY_OK, DOWNLOAD_FAILED, SAVE_FAILED, MD5_FAILED, MYSTERY_FAILED)
    names = ('NEW_OK', 'ALREADY_OK', 'DOWNLOAD_FAILED', 'SAVE_FAILED', 'MD5_FAILED',
             'MYSTERY_FAILED')

def ensure_dir(dirpath):
    if not os.path.exists(dirpath): os.makedirs(dirpath)

def ensure_parent_dir(childpath):
    ensure_dir(os.path.dirname(childpath))

def logmsg(msg, flog=None):
    tmsg = '[%s] %s' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    print '%s' % tmsg
    if flog is not None:
        flog.write('%s\n' % tmsg)
        flog.flush()

def logexc(msg, exc, flog=None):
    logmsg('{} [{}: {}]'.format(msg, exc.__class__.__name__, exc), flog=flog)

def get_birdsnap(imlistpath, outroot):
    ensure_dir(outroot)
    logpath = os.path.join(outroot, 'log.txt')
    with open(logpath, 'at') as flog:
        logmsg('Starting.', flog=flog)
        temproot, imageroot = (os.path.join(outroot, dirname) for dirname in ('temp', 'images'))
        if os.path.exists(temproot):
            logmsg('Removing existing temp directory.', flog=flog)
            shutil.rmtree(temproot)
        for d in (temproot, imageroot): ensure_dir(d)
        logmsg('Reading images list from file {}...'.format(imlistpath), flog=flog)
        images = read_list_of_dicts(imlistpath)
        n_images = len(images)
        logmsg('{} images in list.'.format(n_images), flog=flog)
        results = dict((result, 0) for result in DownloadResult.values)
        successpath, failpath = (os.path.join(outroot, fname)
                                 for fname in ('success.txt', 'fail.txt'))
        with open(successpath, 'wt') as fsuccess, open(failpath, 'wt') as ffail:
            for i_image, image in enumerate(images):
                logmsg('Start image {} of {}: {}'.format(i_image + 1, n_images, image['path']),
                       flog=flog)
                result = download_image(image, temproot, imageroot, flog=flog)
                (fsuccess if result in (DownloadResult.NEW_OK, DownloadResult.ALREADY_OK)
                 else ffail).write('{}\t{}\n'.format(image['url'], image['path']))
                results[result] += 1
                logmsg('Finished image {} of {} with result {}.  Progress is {}.'.format(
                        i_image + 1, n_images, DownloadResult.names[result],
                        ', '.join('{}:{}'.format(DownloadResult.names[k], v)
                                  for k, v in results.items())),
                       flog=flog)
        logmsg('Finished.', flog=flog)

def check_image(image, imagepath):
    if not os.path.exists(imagepath): return False
    with open(imagepath, 'rb') as fin:
        return (hashlib.md5(fin.read()).hexdigest() == image['md5'])

def download_image(image, temproot, imageroot, flog=None):
    # Check existing file.
    try:
        temppath, imagepath = (os.path.join(root, image['path']) for root in (temproot, imageroot))
        if check_image(image, imagepath):
            logmsg('Already have the image and contents are correct.', flog=flog)
            return DownloadResult.ALREADY_OK
        else:
            logmsg('Need to get the image.', flog=flog)
            if os.path.exists(imagepath):
                logmsg('Deleting existing bad file {}.'.format(imagepath), flog=flog)
                os.remove(imagepath)
    except Exception as e:
        logexc('Unexpected exception before attempting download of image {!r}.'.format(image), e,
               flog=flog)
        return DownloadResult.MYSTERY_FAILED
    # GET and save to temp location.
    try:
        r = requests.get(image['url'])
        if r.status_code == 200:
            ensure_parent_dir(temppath)
            with open(temppath, 'wb') as fout:
                for chunk in r.iter_content(1024): fout.write(chunk)
            logmsg('Saved  {}.'.format(temppath), flog=flog)
        else:
            logmsg('Status code {} when requesting {}.'.format(r.status_code, image['url']))
            return DownloadResult.DOWNLOAD_FAILED
    except Exception as e:
        logexc('Unexpected exception when downloading image {!r}.'.format(image), e, flog=flog)
        return DownloadResult.DOWNLOAD_FAILED
    # Check contents.
    try:
        if check_image(image, temppath):
            logmsg('Image contents look good.', flog=flog)
        else:
            logmsg('Image contents are wrong.', flog=flog)
            return DownloadResult.MD5_FAILED
    except Exception as e:
        logexc('Unexpected exception when checking file contents for image {!r}.'.format(image), e,
               flog=flog)
        return DownloadResult.MYSTERY_FAILED
    # Move image to final location.
    try:
        ensure_parent_dir(imagepath)
        os.rename(temppath, imagepath)
    except Exception as e:
        logexc('Unexpected exception when moving file from {} to {} for image {!r}.'.format(
                temppath, imagepath, image), e, flog=flog)
        return DownloadResult.MYSTERY_FAILED
    return DownloadResult.NEW_OK

def read_list_of_dicts(path):
    rows = []
    with open(path, 'r') as fin:
        fieldnames = fin.readline().strip().split('\t')
        for line in fin:
            vals = line.strip().split('\t')
            assert len(vals) == len(fieldnames)
            rows.append(dict(zip(fieldnames, vals)))
        return rows

def testargs():
    imlistpath = '/media/lacie_alpha/thomas/cubirds/data/dist/1.0-rec-debug-20150125-143026/images.txt'
    outroot = '/media/lacie_alpha/thomas/cubirds/data/dist/testout'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download the images of the Birdsnap dataset.')
    parser.add_argument(
        '--output_dir',
        default='download',
        help='directory in which to save the images and output files')
    parser.add_argument(
        '--images-file',
        default='images.txt',
        help='image list file')
    args = parser.parse_args()
    get_birdsnap(args.images_file, args.output_dir)
