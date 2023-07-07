import os,sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from cleanfid import fid
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('img_folder', '~/', """GAN Image Folder""")
flags.DEFINE_string('src_folder', '~/', """Source Image Folder""")
flags.DEFINE_string('dataset_name', 'afhq_cat', """Name of the Dataset""")
flags.DEFINE_string('dataset_split', 'train', """Name of the Dataset""")
flags.DEFINE_string('mode', 'clean', """clean or legacy modes""")
flags.DEFINE_integer('dataset_res', 512, """dataset resultion""")
flags.DEFINE_integer('FID_flag', 1, """dataset resultion""")
flags.DEFINE_integer('KID_flag', 1, """dataset resultion""")
flags.DEFINE_integer('folder_flag', 0, """dataset resultion""")

FLAGS(sys.argv)

FLAGS_dict = FLAGS.flag_values_dict()

# fid_score = fid.compute_fid(FLAGS_dict['img_folder'], dataset_name=FLAGS_dict['dataset_name'], dataset_res=FLAGS_dict['dataset_res'],  mode=FLAGS_dict['mode'], dataset_split=FLAGS_dict['dataset_split'])


print(FLAGS_dict)
if FLAGS_dict['FID_flag']:
	try:
		if FLAGS_dict['folder_flag']:
			raise Exception("Forcibly executing on folder of images")
		else:
			fid_score = fid.compute_fid(FLAGS_dict['img_folder'], dataset_name=FLAGS_dict['dataset_name'], dataset_res=FLAGS_dict['dataset_res'],  mode=FLAGS_dict['mode'], dataset_split=FLAGS_dict['dataset_split'])
	except:
		try:
			fid_score = fid.compute_fid(FLAGS_dict['img_folder'], fdir2=FLAGS_dict['src_folder'])
		except:
			print('FID Computation Failed')
	print('FID: ',fid_score)

if FLAGS_dict['KID_flag']:
	try:
		if FLAGS_dict['folder_flag']:
			raise Exception("Forcibly executing on folder of images")
		else:
			kid_score = fid.compute_kid(FLAGS_dict['img_folder'], dataset_name=FLAGS_dict['dataset_name'], dataset_res=FLAGS_dict['dataset_res'],  mode=FLAGS_dict['mode'], dataset_split=FLAGS_dict['dataset_split'])
	except:
		try:
			kid_score = fid.compute_kid(FLAGS_dict['img_folder'], fdir2=FLAGS_dict['src_folder'])
		except:
			print('KID Computation Failed')
	print('KID: ',kid_score)



