import models as md
import utils as ut

md.Models.XCEPTION.build()
features, labels = md.Models.XCEPTION.get_features(ut.Directory.test)
ut.save_to_file("%s_xcpt" % ut.Directory.test.name, features, labels)
codes, labels = ut.read_file("%s_xcpt" % ut.Directory.test.name)

print("Reading Shape", codes.shape)