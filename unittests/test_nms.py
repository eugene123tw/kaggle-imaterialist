import unittest


class TestNMSMethods(unittest.TestCase):

  def test_import(self):
    from mmdetection.mmdet.ops.nms import nms_cuda, nms_cpu
    from mmdetection.mmdet.ops.nms.soft_nms_cpu import soft_nms_cpu


if __name__ == '__main__':
  unittest.main()