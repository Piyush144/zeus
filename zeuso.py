import tensorflow as tf, sys
import os,time
from socket import *
from random import randint
import MySQLdb
conn = MySQLdb.connect(host= "localhost",
                  user="root",
                  passwd="",
                  db="autorefresh")
x = conn.cursor()

array=[5]
array.append(sys.argv[1])
array.append(sys.argv[2])
array.append(sys.argv[3])
array.append(sys.argv[4])
array.append(sys.argv[5])



# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
 in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
 graph_def = tf.GraphDef()
 graph_def.ParseFromString(f.read())
 _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:    
 while True:
  try:
    rnum=randint(0,9)
    image_path = "pic.jpeg"
    start=time.time()
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
                                     {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
     human_string = label_lines[node_id]
     score = predictions[0][node_id]
     net=(('%s (score = %.5f)' % (human_string, score)))
     for pick in range(len(array)):
      if array[pick] == human_string: 
       if score > 0.2:
        print ("")
        print ("")
        print (net)
        print ("")
        print ("")
        print ("")
        print (time.time()-start)
        fianlname="C:\\wamp64\\www\\sql\\images\\"+"pic"*rnum+".jpeg"
        os.system("copy %s %s" % ("pic.jpeg","C:\\wamp64\\www\\sql\\images\\"))
        os.rename('C:\\wamp64\\www\sql\\images\\pic.jpeg',fianlname)
        try:
            x.execute("""UPDATE autorefresh SET name =%s , address = %s , image_name=%s, prob=%s """,("IP: 121.242.232.134","LOCATION- iitm reseach campus,chennai","images/pic.jpeg",net))
            conn.commit()
        except:
            conn.rollback()

        conn.close()
        os.remove("pic.jpeg")
    os.remove("pic.jpeg")
  except:pass
