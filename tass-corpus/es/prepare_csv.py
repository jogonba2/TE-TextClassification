from xml.dom import minidom


## TRAIN/DEV ##
"""
doc = minidom.parse("intertass_es_dev.xml")
frclases = open("intertass_es_dev_gold.tsv", "r", encoding="utf8")
fw = open("dev.csv", "w", encoding="utf8")
fw.write("ID\tTWEET\tCLASS\n")
tweets = doc.getElementsByTagName("tweet")
ids = [tweets[i].getElementsByTagName("tweetid")[0].firstChild.data for i in range(len(tweets))]
texts = [tweets[i].getElementsByTagName("content")[0].firstChild.data for i in range(len(tweets))]
classes = [line.strip().split("\t")[-1] for line in frclases.readlines()]


for i in range(len(ids)):
    fw.write(ids[i] + "\t" + texts[i] + "\t" + classes[i] + "\n")

fw.close()
frclases.close()
"""


## TEST ##
doc = minidom.parse("intertass_es_test.xml")
fw = open("test.csv", "w", encoding="utf8")
fw.write("ID\tTWEET\tCLASS\n")
tweets = doc.getElementsByTagName("tweet")
ids = [tweets[i].getElementsByTagName("tweetid")[0].firstChild.data for i in range(len(tweets))]
texts = [tweets[i].getElementsByTagName("content")[0].firstChild.data for i in range(len(tweets))]
classes = ["?" for i in range(len(ids))]


for i in range(len(ids)):
    fw.write(ids[i] + "\t" + texts[i] + "\t" + classes[i] + "\n")

fw.close()
