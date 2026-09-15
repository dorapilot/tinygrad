import ctypes, errno, mmap, os, struct, sys, tempfile, unittest
from unittest.mock import Mock, patch
from tinygrad.runtime.autogen import msm_drm

def ioctl_number(ioctl):
  direction, base, number, struct_type = ioctl.args
  return direction << 30 | ctypes.sizeof(struct_type) << 16 | base << 8 | number

class TestMSMDRMUAPI(unittest.TestCase):
  def test_layouts(self):
    layouts = {
      msm_drm.struct_drm_msm_param: (24, (0, 4, 8, 16, 20)),
      msm_drm.struct_drm_msm_gem_new: (16, (0, 8, 12)),
      msm_drm.struct_drm_msm_gem_info: (24, (0, 4, 8, 16, 20)),
      msm_drm.struct_drm_msm_gem_submit_cmd: (32, (0, 4, 8, 12, 16, 20, 24, 24)),
      msm_drm.struct_drm_msm_gem_submit_bo: (16, (0, 4, 8)),
      msm_drm.struct_drm_msm_gem_submit: (72, (0, 4, 8, 12, 16, 24, 32, 36, 40, 48, 56, 60, 64, 68)),
    }
    for struct_type, (size, offsets) in layouts.items():
      self.assertEqual((ctypes.sizeof(struct_type), tuple(x[2] for x in struct_type._real_fields_)), (size, offsets))
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_GEM_CLOSE), 0x40086409)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GET_PARAM), 0xC0186440)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT), 0xC0486446)

@unittest.skipIf(sys.platform == "win32", "QCOM is not supported on Windows")
class TestMSMInterface(unittest.TestCase):
  def test_address_lookup_scaling(self):
    from tinygrad.runtime.ops_qcom import MSMAllocation, MSMIface
    class CountedAllocation(MSMAllocation):
      reads = 0
      def __getattribute__(self, name):
        if name == 'iova': type(self).reads += 1
        return super().__getattribute__(name)
    iface = object.__new__(MSMIface)
    iface.allocations = {i:CountedAllocation(i, (i+1)*4096, 2048, 0, 0) for i in range(1024)}
    for i in range(0, 1024, 4):
      self.assertEqual(iface._allocation((i+1)*4096 + 32, 64).handle, i)
    self.assertLess(CountedAllocation.reads, 1024 * 20, 'Address resolution rescans all live allocations for each buffer')

  def test_free_retries_cleanup(self):
    from tinygrad.device import BufferStorage
    from tinygrad.runtime.ops_qcom import MSMAllocation, MSMIface
    iface = object.__new__(MSMIface)
    allocation = MSMAllocation(7, 0x10000000, 64, mmap.PAGESIZE, 0x20000000)
    storage = BufferStorage(allocation.iova, allocation)
    iface.fd, iface.allocations = Mock(), {7: allocation}
    iface.fd.munmap.side_effect = [-1, 0]
    with patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE', side_effect=[OSError(errno.EIO, 'close failed'), None]) as close:
      with self.assertRaisesRegex(RuntimeError, 'unmap'): iface.free(storage)
      close.assert_not_called()
      with self.assertRaisesRegex(OSError, 'close failed'): iface.free(storage)
      iface.free(storage)
      self.assertEqual(iface.fd.munmap.call_count, 2)
      self.assertEqual(close.call_count, 2)
      self.assertEqual(iface.allocations, {})

  def test_import_alias_lifetime(self):
    from tinygrad.runtime.ops_qcom import MSMIface
    from tinygrad.runtime.support.hcq import FileIOInterface
    iface = object.__new__(MSMIface)
    iface.fd, iface.allocations = Mock(), {}
    iface.fd.munmap.side_effect = FileIOInterface.munmap
    memory = (ctypes.c_ubyte * mmap.PAGESIZE)()
    with self.assertRaisesRegex(RuntimeError, 'not allocated'): iface._allocation(0x10000000, 64)
    with tempfile.TemporaryFile() as source, \
         patch.object(msm_drm, 'DRM_IOCTL_PRIME_FD_TO_HANDLE', return_value=Mock(handle=7)) as prime, \
         patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_INFO', return_value=Mock(value=0x10000000)), \
         patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE') as close:
      source.truncate(mmap.PAGESIZE)
      first = iface.map(ctypes.addressof(memory), 64, source.fileno())
      self.assertIs(iface._allocation(first.buf, 64), first.meta)
      second = iface.map(ctypes.addressof(memory) + 32, 32, source.fileno(), 32)
      self.assertEqual((first.buf, second.buf), (0x10000000, 0x10000020))
      self.assertIs(first.meta, second.meta)
      self.assertIsNone(first.host)
      with self.assertRaisesRegex(ValueError, 'exceeds DMA-BUF'):
        iface.map(ctypes.addressof(memory), mmap.PAGESIZE + 1, source.fileno())
      self.assertEqual(prime.call_count, 2)
      iface.free(first)
      close.assert_not_called()
      self.assertIs(iface._allocation(second.buf, 32), second.meta)
      iface.free(second)
      close.assert_called_once_with(iface.fd, handle=7)
      self.assertEqual(iface.allocations, {})
      self.assertEqual(os.fstat(source.fileno()).st_size, mmap.PAGESIZE)
      iface.fd.munmap.assert_called_once()
      with self.assertRaisesRegex(RuntimeError, 'not allocated'): iface._allocation(second.buf, 32)

  def test_allocation_and_submit(self):
    from tinygrad.runtime.ops_qcom import MSMAllocation, MSMIface
    memory = [(ctypes.c_ubyte * mmap.PAGESIZE)() for _ in range(2)]
    fd = Mock()
    fd.mmap.side_effect, fd.munmap.return_value = [ctypes.addressof(x) for x in memory], 0
    iface = object.__new__(MSMIface)
    iface.dev, iface.fd, iface.allocations = Mock(error_state=None), fd, {}
    iovas = {7:0x10000000, 9:0x20000000}

    def gem_info(_fd, handle, info):
      return Mock(value=iovas[handle] if info == msm_drm.MSM_INFO_GET_IOVA else handle * mmap.PAGESIZE)

    with (
      patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_NEW', side_effect=[Mock(handle=7), Mock(handle=9)]) as gem_new,
      patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_INFO', side_effect=gem_info),
    ):
      command = iface.alloc(17)
      self.assertIs(iface._allocation(command.buf + 4, 8), command.meta)
      data = iface.alloc(32)

    self.assertIsInstance(command.meta, MSMAllocation)
    self.assertEqual((command.meta.size, command.meta.mapped_size), (17, mmap.PAGESIZE))
    self.assertEqual([call.kwargs['flags'] for call in gem_new.call_args_list], [msm_drm.MSM_BO_WC, msm_drm.MSM_BO_WC])

    buffers = [(data.buf, 32), (data.buf + 4, 8), (data.buf, 32)]
    submit, bos, cmds = iface.prepare_submit(command.buf + 4, 8, buffers)
    read_write = msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE
    self.assertEqual((submit.nr_bos, submit.queueid), (2, 0))
    self.assertEqual([(bo.flags, bo.handle, bo.presumed) for bo in bos], [
      (msm_drm.MSM_SUBMIT_BO_READ, 7, 0x10000000),
      (read_write, 9, 0x20000000),
    ])
    self.assertEqual((cmds[0].submit_idx, cmds[0].submit_offset, cmds[0].size), (0, 4, 8))
    with self.assertRaisesRegex(RuntimeError, "not allocated"): iface.prepare_submit(command.buf + 16, 4, [])
    with patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE'):
      iface.free(data)
    with self.assertRaisesRegex(RuntimeError, 'not allocated'): iface.prepare_submit(command.buf, 4, [(data.buf, 4)])
    self.assertIs(iface._allocation(command.buf, 4), command.meta)

@unittest.skipIf(sys.platform == "win32", "QCOM is not supported on Windows")
class TestMSMReplay(unittest.TestCase):
  def setUp(self):
    from tinygrad import Device
    from tinygrad.helpers import Context
    from tinygrad.runtime.support.hcq import FileIOInterface
    from tinygrad.runtime.ops_qcom import MSMIface, QCOMDevice
    self.enterContext(Context(DEV="MSM+QCOM:IR3"))
    self.memory, self.submitted = {}, []
    fd = Mock(spec=FileIOInterface)
    fd.ioctl.return_value = 0
    def unmap(addr, size):
      return 0 if any(addr == ctypes.addressof(x) for x in self.memory.values()) else FileIOInterface.munmap(addr, size)
    fd.munmap.side_effect = unmap
    def new(_fd, size, flags):
      handle = len(self.memory) + 1
      self.memory[handle] = (ctypes.c_ubyte * size)()
      return Mock(handle=handle)
    def info(_fd, handle, info): return Mock(value=handle * (1 << 28) if info == msm_drm.MSM_INFO_GET_IOVA else handle * mmap.PAGESIZE)
    fd.mmap.side_effect = lambda _addr, _size, _prot, _flags, offset: ctypes.addressof(self.memory[offset // mmap.PAGESIZE])
    self.enterContext(patch('tinygrad.runtime.ops_qcom.glob.glob', return_value=['/dev/dri/renderD128']))
    self.enterContext(patch('tinygrad.runtime.ops_qcom._open_msm_render_node', return_value=(fd, 0x06030002)))
    self.enterContext(patch.object(msm_drm, 'DRM_IOCTL_MSM_GET_PARAM', return_value=Mock(value=0)))
    self.enterContext(patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_NEW', side_effect=new))
    self.enterContext(patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_INFO', side_effect=info))
    self.enterContext(patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE'))
    self.enterContext(patch.object(QCOMDevice, '_select_iface', lambda dev, _: MSMIface(dev, 0)))
    self.dev = QCOMDevice('QCOM')
    self.dev.rtalloc_size = 1 << 20
    getitem = type(Device).__getitem__
    self.enterContext(patch.object(type(Device), '__getitem__', lambda obj, key: self.dev if key == 'QCOM' else getitem(obj, key)))
    self.submit = self.enterContext(patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_SUBMIT', side_effect=self.record_submit))

  def record_submit(self, _fd, **kwargs):
    req = kwargs['__payload']
    bos = (msm_drm.struct_drm_msm_gem_submit_bo * req.nr_bos).from_address(req.bos)
    cmd = msm_drm.struct_drm_msm_gem_submit_cmd.from_address(req.cmds)
    self.assertEqual((req.queueid, req.nr_cmds, req.flags), (0, 1, msm_drm.MSM_PIPE_3D0))
    self.assertEqual(cmd.type, msm_drm.MSM_SUBMIT_CMD_BUF)
    allocation = self.dev.iface.allocations[bos[cmd.submit_idx].handle]
    self.submitted.append(([bo.handle for bo in bos], bytes(self.memory[allocation.handle])))
    # This is an ioctl contract test, with no GPU execution. Complete the mocked timeline so host replay can proceed.
    self.dev.timeline.host.view(fmt='Q')[0] = self.dev.timeline.host.view(fmt='Q')[1]

  def tearDown(self):
    self.dev.iface.submit_error = None
    self.dev.timeline.host.view(fmt='Q')[0] = self.dev.timeline.host.view(fmt='Q')[1]

  def compile_signal(self):
    from tinygrad import dtypes
    from tinygrad.uop.ops import UOp, Ops, KernelInfo
    from tinygrad.runtime.support.hcq2 import make_submit, lower_call, HCQInfo, hcq_link
    from tinygrad.engine.realize import lower_and_compile
    buf = UOp.param(0, dtypes.float32, 257, device='QCOM')
    submit = make_submit(UOp(Ops.INS, arg=('store', dtypes.void), src=(buf, UOp.const(1, dtypes.uint64))),
                         devs=('QCOM',), queue='COMPUTE:0')
    call = UOp.sink(submit, arg=KernelInfo('submit_test')).call(buf, aux=HCQInfo(('QCOM',)))
    return hcq_link(lower_and_compile(UOp(Ops.LINEAR, src=(lower_call(call),))), allow_cache=False)

  def test_import_tensor_cpu_access(self):
    from tinygrad import Tensor, dtypes
    from tinygrad.device import Buffer
    from tinygrad.runtime.autogen import dma_buf
    with tempfile.TemporaryFile() as source, \
         patch.object(msm_drm, 'DRM_IOCTL_PRIME_FD_TO_HANDLE', return_value=Mock(handle=4096)), \
         patch.object(dma_buf, 'DMA_BUF_IOCTL_SYNC') as sync:
      source.truncate(96)
      memory = mmap.mmap(source.fileno(), 96)
      memory[:] = bytes(range(96))
      tensor = Tensor.from_blob(ctypes.addressof(ctypes.c_ubyte.from_buffer(memory)) + 32, (64,), fd=source.fileno(), offset=32,
                                dtype=dtypes.uint8, device='QCOM')
      buf = tensor.uop.buffer
      try:
        sync.side_effect = [OSError(errno.EINTR, 'interrupted'), None, None]
        self.assertEqual(buf.numpy().tolist(), list(range(32, 96)))
        self.assertEqual([c.kwargs['flags'] for c in sync.call_args_list], [dma_buf.DMA_BUF_SYNC_READ] * 2 +
                         [dma_buf.DMA_BUF_SYNC_READ | dma_buf.DMA_BUF_SYNC_END])
        sync.reset_mock(side_effect=True)
        buf.copy_from(Buffer('PYTHON', 64, dtypes.uint8, initial_value=bytes(reversed(range(64)))))
        self.assertEqual(list(memory[:]), list(range(32)) + list(reversed(range(64))))
        self.assertEqual([c.kwargs['flags'] for c in sync.call_args_list],
                         [dma_buf.DMA_BUF_SYNC_WRITE, dma_buf.DMA_BUF_SYNC_WRITE | dma_buf.DMA_BUF_SYNC_END])
        sync.side_effect = OSError(errno.EIO, 'sync failed')
        with self.assertRaises(OSError): buf.numpy()
      finally:
        buf.deallocate()
        memory.close()

  def test_cached_owned_import_alias(self):
    from tinygrad import dtypes
    from tinygrad.device import Buffer
    original = Buffer('QCOM', 64, dtypes.uint8, preallocate=True)
    allocation, ptr = original.meta, original.host.addr
    original.deallocate()
    with tempfile.TemporaryFile() as source, \
         patch.object(msm_drm, 'DRM_IOCTL_PRIME_FD_TO_HANDLE', return_value=Mock(handle=allocation.handle)), \
         patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE') as close:
      source.truncate(64)
      imported = Buffer('QCOM', 64, dtypes.uint8).allocate(external_ptr=ptr, external_fd=source.fileno())
      self.assertEqual(allocation.references, 1)
      self.assertFalse(any(s.meta is allocation for cached in self.dev.allocator.cache.values() for s in cached))
      close.assert_not_called()
      imported.deallocate()
      close.assert_called_once_with(self.dev.iface.fd, handle=allocation.handle)
      self.assertNotIn(allocation.handle, self.dev.iface.allocations)

  def test_import_survives_released_alias_mapping(self):
    from tinygrad import Tensor, dtypes
    from tinygrad.runtime.autogen import dma_buf
    with tempfile.TemporaryFile() as source, \
         patch.object(msm_drm, 'DRM_IOCTL_PRIME_FD_TO_HANDLE', return_value=Mock(handle=4096)), \
         patch.object(dma_buf, 'DMA_BUF_IOCTL_SYNC'):
      source.truncate(mmap.PAGESIZE)
      with mmap.mmap(source.fileno(), mmap.PAGESIZE) as first_map, mmap.mmap(source.fileno(), mmap.PAGESIZE) as second_map:
        second_map[:96] = bytes(range(96))
        first_ptr, second_ptr = [ctypes.addressof(ctypes.c_ubyte.from_buffer(mapping)) for mapping in (first_map, second_map)]
        first = Tensor.from_blob(first_ptr, (64,), fd=source.fileno(), dtype=dtypes.uint8, device='QCOM').uop.buffer
        second = Tensor.from_blob(second_ptr + 32, (64,), fd=source.fileno(), offset=32, dtype=dtypes.uint8, device='QCOM').uop.buffer
        first.deallocate()
        first_map.close()
        memmove = ctypes.memmove
        def checked_copy(dest, src, size):
          self.assertFalse(first_ptr <= src < first_ptr + mmap.PAGESIZE, 'CPU copy uses a released mapping')
          return memmove(dest, src, size)
        try:
          with patch('tinygrad.runtime.ops_qcom.ctypes.memmove', side_effect=checked_copy):
            self.assertEqual(second.numpy().tolist(), list(range(32, 96)))
        finally:
          second.deallocate()

  def test_owned_storage_syncs_after_reimport(self):
    from tinygrad import dtypes
    from tinygrad.device import Buffer
    from tinygrad.runtime.autogen import dma_buf
    original = Buffer('QCOM', 64, dtypes.uint8, preallocate=True)
    view = original.view(32, dtypes.uint8, 32).ensure_allocated()
    allocation, ptr = original.meta, original.host.addr
    with tempfile.TemporaryFile() as source, \
         patch.object(msm_drm, 'DRM_IOCTL_PRIME_FD_TO_HANDLE', return_value=Mock(handle=allocation.handle)), \
         patch.object(dma_buf, 'DMA_BUF_IOCTL_SYNC') as sync:
      source.truncate(64)
      imported = Buffer('QCOM', 64, dtypes.uint8).allocate(external_ptr=ptr, external_fd=source.fileno())
      try:
        for buf in (original, view):
          with self.subTest(size=buf.nbytes):
            sync.reset_mock()
            buf.numpy()
            self.assertEqual([c.kwargs['flags'] for c in sync.call_args_list],
                             [dma_buf.DMA_BUF_SYNC_READ, dma_buf.DMA_BUF_SYNC_READ | dma_buf.DMA_BUF_SYNC_END])
            sync.reset_mock()
            data = bytes(reversed(range(buf.nbytes)))
            buf.copy_from(Buffer('PYTHON', buf.nbytes, dtypes.uint8, initial_value=data))
            self.assertEqual([c.kwargs['flags'] for c in sync.call_args_list],
                             [dma_buf.DMA_BUF_SYNC_WRITE, dma_buf.DMA_BUF_SYNC_WRITE | dma_buf.DMA_BUF_SYNC_END])
            self.assertEqual(buf.as_memoryview(allow_zero_copy=True).tobytes(), data)
      finally:
        view.deallocate()
        imported.deallocate()
        original.deallocate()

  def test_replacement_views(self):
    from tinygrad import dtypes
    from tinygrad.device import Buffer
    from tinygrad.uop.ops import UOp
    from tinygrad.engine.realize import run_linear
    from test.helpers import call_is_hcq
    linear = self.compile_signal()
    self.assertTrue(any(call_is_hcq(c) for c in linear.src))
    for offset in (0, 68):
      with self.subTest(offset=offset):
        bases = [Buffer('QCOM', 274, dtypes.float32, preallocate=True) for _ in range(2)]
        bufs = [b.view(257, dtypes.float32, offset).ensure_allocated() for b in bases]
        self.assertNotEqual(bufs[0]._buf, bufs[1]._buf)
        for i in range(4):
          run_linear(linear, input_uops=[UOp.from_buffer(bufs[i % 2])], jit=True)
          handles, command = self.submitted[-1]
          active, other = bufs[i % 2], bufs[1 - i % 2]
          self.assertIn(active.meta.handle, handles)
          self.assertNotIn(other.meta.handle, handles)
          self.assertIn(struct.pack('<Q', active._buf), command)
          self.assertNotIn(struct.pack('<Q', other._buf), command)

  def test_submit_retries_transient_errors(self):
    from tinygrad import dtypes
    from tinygrad.device import Buffer
    from tinygrad.uop.ops import UOp
    from tinygrad.engine.realize import run_linear
    linear = self.compile_signal()
    buf = Buffer('QCOM', 257, dtypes.float32, preallocate=True)
    errors = iter([errno.EINTR, errno.EAGAIN, None, errno.EINTR, None])
    def submit(fd, **kwargs):
      if (error := next(errors)) is not None: raise OSError(error, 'submit interrupted')
      return self.record_submit(fd, **kwargs)
    self.submit.side_effect = submit
    for _ in range(2):
      run_linear(linear, input_uops=[UOp.from_buffer(buf)], jit=True)
      self.dev.synchronize()
    self.assertEqual(len(self.submitted), 2)
    self.assertEqual(self.submit.call_count, 5)
    self.assertIsNone(self.dev.iface.submit_error)

  def test_submit_error_propagates(self):
    from tinygrad import dtypes
    from tinygrad.device import Buffer
    from tinygrad.uop.ops import UOp
    from tinygrad.engine.realize import run_linear
    linear = self.compile_signal()
    buf = Buffer('QCOM', 257, dtypes.float32, preallocate=True)
    error = OSError(22, 'submit rejected')
    self.submit.side_effect = error
    run_linear(linear, input_uops=[UOp.from_buffer(buf)], jit=True)
    with self.assertRaises(OSError) as caught: self.dev.synchronize()
    self.assertIs(caught.exception, error)
    run_linear(linear, input_uops=[UOp.from_buffer(buf)], jit=True) # a latched error stops later submissions
    with self.assertRaises(OSError): self.dev.synchronize()
    self.assertEqual(self.submit.call_count, 1)

if __name__ == '__main__':
  unittest.main()
