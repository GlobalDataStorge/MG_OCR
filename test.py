# coding=utf-8
import fetchpic_from_baidu as fetch

print "fuck.."

p = fetch.downloadFromBaidu("金毛", 30, (128, 128))
p.close()
p.join()
