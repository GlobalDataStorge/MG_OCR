# coding: utf-8
import urllib2
import json
import multiprocessing
import datetime
from multiprocessing import Pool

searchUrl = "http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={keyword}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&word={keyword}&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&fr=&pn={pageNum}&rn=30"  # &gsm=b4&1499849456929="

pool = Pool(20)
size = (128, 128)

send_headers = {
    'Host': 'img5.imgtn.bdimg.com',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6',
    'If-Modified-Since': 'Thu, 01 Jan 1970 00:00:00 GMT',
    'Referer': 'img5.imgtn.bdimg.com/it/',
    'X-Requested-With': 'XMLHttpRequest'
}


def resize(b_data, name, size=(128, 128)):
    from PIL import Image
    import cStringIO as StringIO
    try:
        stream = StringIO.StringIO(b_data)
        im = Image.open(stream)
        im.thumbnail(size)
        im.save(name, "JPEG")
    except IOError:
        print("cannot create thumbnail for", name)


def download(data):
    # resp = urllib2.urlopen(data['thumbURL'].decode("utf-8"))
    # resp = urllib2.urlopen("http://img5.imgtn.bdimg.com/it/u=2618620045,2641964444&fm=26&gp=0.jpg")
    if data.has_key('thumbURL'):
        try:
            url = data['thumbURL']
            req = urllib2.Request(url, headers=send_headers)
            resp = urllib2.urlopen(req)
            raw = resp.read()
            import hashlib
            global size
            filename = url.split("/")[-1]
            # resize(raw, "image/" + str(hashlib.md5(url).hexdigest()) + "_" + filename, size)
            resize(raw, "image/" + filename, size)
        except Exception, e:
            pass
            # print "url"
            # print data['thumbURL']
            # print e


pool = Pool(4)


def downloadFromBaidu(keyworld, num, sz):
    pages = num / 30 + 1
    global size
    size = sz
    for i in range(0, pages):
        response = urllib2.urlopen(searchUrl.format(keyword=keyworld, pageNum=i * 30), timeout=10)
        data = response.read()
        try:
        # data = urllib2.unquote(data)
            obj = json.loads(data)

            global pool
            pool.map_async(download, obj['data'])
        except Exception,e:
            print e
    return pool


if __name__ == "__main__":
    p = downloadFromBaidu('穿衣搭配', 50000, (128, 128))
    p.close()
    p.join()

    # response = urllib2.urlopen(searchUrl.format(keyword='搭配',pageNum='30'), timeout=10)
    # data = response.read()
    # obj = json.loads(data)
    #
    # pool = Pool(4)
    # pool.map(download,obj['data'])
    # pool.apply()
    # pool.join()

    # req = urllib2.Request("http://img5.imgtn.bdimg.com/it/u=2618620045,2641964444&fm=26&gp=0.jpg", headers=send_headers)
    # resp = urllib2.urlopen(req)
    # raw = resp.read()
    # with open("image/" + str(datetime.datetime.now()) + ".jpg", 'wb') as fp:
    #    fp.write(raw)


    # print urllib2.urlopen("http://www.baidu.com").read()
