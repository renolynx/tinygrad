from __future__ import annotations
import time, functools, array, tinygrad.runtime.autogen.nv_regs
from tinygrad.helpers import getenv, DEBUG, getbits, round_up
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.support.memory import TLSFAllocator, MemoryManager, AddrSpace
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_FLCN_COT, NV_GSP
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.runtime.support.hcq import MMIOInterface

NV_DEBUG = getenv("NV_DEBUG", 0)
NV_PT_SHADOW_VERIFY = getenv("NV_PT_SHADOW_VERIFY", 0)   # patch #30 self-check
NV_PT_SHADOW = getenv("NV_PT_SHADOW", 1)                 # patch #30 off switch (NV_PT_SHADOW=0 = upstream behaviour)

class NVReg:
  def __init__(self, nvdev, base, off, fields=None): self.nvdev, self.base, self.off, self.fields = nvdev, base, off, fields

  def __getitem__(self, idx:int): return NVReg(self.nvdev, self.base, self.off(idx), fields=self.fields)

  def add_field(self, name:str, start:int, end:int): self.fields[name] = (start, end)
  def with_base(self, base:int): return NVReg(self.nvdev, base + self.base, self.off, self.fields)

  def read(self): return self.nvdev.rreg(self.base + self.off)
  def read_bitfields(self) -> dict[str, int]: return self.decode(self.read())

  def write(self, _ini_val:int=0, **kwargs): self.nvdev.wreg(self.base + self.off, _ini_val | self.encode(**kwargs))

  def update(self, **kwargs): self.write(self.read() & ~self.mask(*kwargs.keys()), **kwargs)

  def mask(self, *names):
    return functools.reduce(int.__or__, ((((1 << (self.fields[nm][1]-self.fields[nm][0] + 1)) - 1) << self.fields[nm][0]) for nm in names), 0)

  def encode(self, **kwargs) -> int: return functools.reduce(int.__or__, (value << self.fields[name][0] for name,value in kwargs.items()), 0)
  def decode(self, val: int) -> dict: return {name:getbits(val, start, end) for name,(start,end) in self.fields.items()}

class NVPageTableEntry:
  # ******** LOCAL PATCH #30 (2026-09-04): host-side shadow of every page-table page ********
  # self.entries is a view into device vram. On the USB4/TinyGPU stack that view is a RemotePCIDevice
  # MMIO window, so reading ONE 8-byte entry is a synchronous socket round trip to TinyGPU.app.
  # map_range/unmap_range read every PTE they touch (the "already mapped" inspection pass, the
  # unmap asserts, and _try_free_pt's 512-entry emptiness scan), which measured 650k reads per 20s
  # = 35% of the wall clock of a SeedVR2 2048px DiT pass. Nothing but this process ever writes these
  # pages - the MMU only reads them - so a write-through host copy is exact. Reads come from the
  # shadow; the first touch of a page costs ONE bulk read of the whole 4 KB instead of up to 512.
  # The shadow is dropped when the page is freed or zeroed (see MemoryManager.palloc/pfree).
  def __init__(self, nvdev, paddr, lv):
    self.nvdev, self.paddr, self.lv, self.entries = nvdev, paddr, lv, nvdev.vram.view(paddr, 0x1000, fmt='Q')
    try: shadows = nvdev._pt_shadow
    except AttributeError: shadows = nvdev._pt_shadow = {}
    if not NV_PT_SHADOW: self.shadow = self.entries; return
    if (sh:=shadows.get(paddr)) is None: shadows[paddr] = sh = array.array('Q', self.entries[:])
    self.shadow = sh

  def _is_dual_pde(self) -> bool: return self.lv == self.nvdev.mm.level_cnt - 2

  def set_entry(self, entry_id:int, paddr:int, table=False, uncached=False, aspace=AddrSpace.PHYS, snooped=False, frag=0, valid=True):
    if not table:
      x = self.nvdev.pte_t.encode(valid=valid, address_sys=paddr >> 12, aperture=2 if aspace is AddrSpace.SYS else 0, kind=6,
        **({'pcf': int(uncached)} if self.nvdev.mmu_ver == 3 else {'vol': uncached}))
    else:
      pde = self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t
      small, sys = ("_small" if self._is_dual_pde() else ""), "" if self.nvdev.mmu_ver == 3 else "_sys"
      x = pde.encode(is_pte=False, **{f'aperture{small}': 1 if valid else 0, f'address{small}{sys}': paddr >> 12},
        **({f'pcf{small}': 0b10} if self.nvdev.mmu_ver == 3 else {'no_ats': 1}))

    if self._is_dual_pde():
      self.entries[2*entry_id], self.entries[2*entry_id+1] = x & 0xffffffffffffffff, x >> 64
      if self.shadow is not self.entries: self.shadow[2*entry_id], self.shadow[2*entry_id+1] = x & 0xffffffffffffffff, x >> 64   # patch #30
    elif self.shadow is not self.entries: self.entries[entry_id] = self.shadow[entry_id] = x
    else: self.entries[entry_id] = x

  def entry(self, entry_id:int) -> int:   # patch #30: served from the host shadow, never the device
    v = (self.shadow[2*entry_id+1]<<64) | self.shadow[2*entry_id] if self._is_dual_pde() else self.shadow[entry_id]
    if NV_PT_SHADOW_VERIFY:   # NV_PT_SHADOW_VERIFY=1: read the device too and prove the shadow never diverges
      d = (self.entries[2*entry_id+1]<<64) | self.entries[2*entry_id] if self._is_dual_pde() else self.entries[entry_id]
      assert d == v, f"pt shadow mismatch pt={self.paddr:#x} lv={self.lv} entry={entry_id}: shadow={v:#x} device={d:#x}"
    return v

  def read_fields(self, entry_id:int) -> dict:
    if self.is_page(entry_id): return self.nvdev.pte_t.decode(self.entry(entry_id))
    return (self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t).decode(self.entry(entry_id))

  def is_page(self, entry_id) -> bool: return (self.entry(entry_id) & 1 == 1) if self.lv < self.nvdev.mm.level_cnt - 1 else True
  def supports_huge_page(self, paddr:int): return self.lv >= self.nvdev.mm.level_cnt - 3 and paddr % self.nvdev.mm.pte_covers[self.lv] == 0

  def valid(self, entry_id):
    if not NV_PT_SHADOW:   # upstream path, kept so NV_PT_SHADOW=0 reverts the whole patch
      if self.is_page(entry_id): return self.read_fields(entry_id)['valid']
      return self.read_fields(entry_id)['aperture_small' if self._is_dual_pde() else 'aperture'] != 0
    # patch #30: one entry read and one getbits instead of two reads and a full decode() dict of every
    # field - map/unmap call this millions of times per image and the dict was pure garbage.
    v = self.entry(entry_id)
    if (v & 1 == 1) if self.lv < self.nvdev.mm.level_cnt - 1 else True: return getbits(v, *self.nvdev.pte_t.fields['valid'])
    dual = self._is_dual_pde()
    return getbits(v, *(self.nvdev.dual_pde_t if dual else self.nvdev.pde_t).fields['aperture_small' if dual else 'aperture']) != 0

  def address(self, entry_id:int) -> int:
    small, sys = ("_small" if self._is_dual_pde() else ""), "_sys" if self.nvdev.mmu_ver == 2 or self.lv == self.nvdev.mm.level_cnt - 1 else ""
    return self.read_fields(entry_id)[f'address{small}{sys}'] << 12

class NVMemoryManager(MemoryManager):
  va_allocator = TLSFAllocator((1 << 44), base=0x1000000000) # global for all devices.

  def on_range_mapped(self): self.dev.NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE.write((1 << 0) | (1 << 1) | (1 << 6) | (1 << 31))

class NVDev:
  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev, self.devfmt, self.mmio = pci_dev, pci_dev.pcibus, pci_dev.map_bar(0, fmt='I')

    self.smi_dev, self.is_booting, self.is_err_state = False, True, False
    self._early_ip_init()
    self._early_mmu_init()

    # No booting state, gsp client is reinited every run.
    self.is_booting = False

    for ip in [self.flcn, self.gsp]: ip.init_sw()
    for ip in [self.flcn, self.gsp]: ip.init_hw()

  def fini(self):
    for ip in [self.gsp, self.flcn]: ip.fini_hw()

  def reg(self, reg:str) -> NVReg: return self.__dict__[reg]
  def wreg(self, addr:int, value:int):
    self.mmio[addr // 4] = value
    if NV_DEBUG >= 4: print(f"wreg: {hex(addr)} = {hex(value)}")
  def rreg(self, addr:int) -> int: return self.mmio[addr // 4]

  def _early_ip_init(self):
    self.reg_names:set[str] = set()
    self.reg_offsets:dict[str, tuple[int, int]] = {}

    self.include("nv_ref", "")
    self.include("dev_fb", "tu102")
    self.include("dev_gc6_island", "ga102")

    if self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read() != 0:
      self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) & ~pci.PCI_COMMAND_MASTER, 2)
      if DEBUG >= 2: print(f"nv {self.devfmt}: WPR2 is up. Issuing a full reset.", flush=True)
      self.pci_dev.reset()
      time.sleep(0.1) # wait until device can respond again

    self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.chip_id = self.reg("NV_PMC_BOOT_0").read()
    self.chip_details = self.reg("NV_PMC_BOOT_42").read_bitfields()
    self.chip_name = {0x17: "GA1", 0x19: "AD1", 0x1b: "GB2"}[self.chip_details['architecture']] + f"{self.chip_details['implementation']:02d}"
    self.fw_name = {"GB2": "gb202", "AD1": "ad102", "GA1": "ga102"}[self.chip_name[:3]]
    self.mmu_ver, self.fmc_boot = (3, True) if self.chip_details['architecture'] >= 0x1a else (2, False)

    self.flcn:NV_FLCN|NV_FLCN_COT = NV_FLCN_COT(self) if self.fmc_boot else NV_FLCN(self)
    self.gsp:NV_GSP = NV_GSP(self)

    self.flcn.wait_for_reset()

  def _early_mmu_init(self):
    self.include("dev_vm", "tu102")

    # MMU Init
    self.include("dev_mmu", "gh100" if self.mmu_ver == 3 else "tu102")
    self.pte_t, self.pde_t, self.dual_pde_t = [self.__dict__[name] for name in [f'NV_MMU_VER{self.mmu_ver}_PTE', f'NV_MMU_VER{self.mmu_ver}_PDE',
                                                                                f'NV_MMU_VER{self.mmu_ver}_DUAL_PDE']]

    self.vram_size = self.reg("NV_PGC6_AON_SECURE_SCRATCH_GROUP_42").read() << 20

    self.vram, self.mmio = self.pci_dev.map_bar(1), self.pci_dev.map_bar(0, fmt='I')
    self.large_bar = self.vram.nbytes >= self.vram_size

    # UVM depth   HW level                            VA bits
    # 0           PDE4                                56:56 (hopper+)
    # 1           PDE3                                55:47
    # 2           PDE2                                46:38
    # 3           PDE1 (or 512M PTE)                  37:29
    # 4           PDE0 (dual 64k/4k PDE, or 2M PTE)   28:21
    # 5           PTE_64K / PTE_4K                    20:16 / 20:12
    bits, shifts = (56, [12, 21, 29, 38, 47, 56]) if self.mmu_ver == 3 else (48, [12, 21, 29, 38, 47])

    # Tail vram reserved for falcon structs and for the GSP firmware.
    #
    # 64 MB (the previous value) is not enough. NV_GSP.init_wpr_meta (support/nv/ip.py) carves the *top* of vram for the GSP: a 1 MB VGA
    # workspace, 1 MB of FRTS, the bootloader, the radix3 GSP ELF, and gspFwHeapSize = 0x8100000 (129 MB), with gspFwRsvdStart marking the
    # bottom of the whole block. Measured on ad102 (RTX 4080 Super) gspFwRsvdStart lands 194 MB below the top of vram, so a 64 MB reservation
    # leaves ~130 MB of GSP-owned memory inside pa_allocator, free to be handed out as tensor storage. A shader writing there takes an MMU
    # NV_PFAULT_FAULT_TYPE_REGION_VIOLATION, which reaches the user as the bare "Device fault detected" from PCIIface.sleep.
    #
    # The exact sizes are firmware-dependent and are not known this early (the images have not been fetched yet), so reserve a bound that
    # covers them. 256 MB is 1.5% of a 16 GB card.
    self.mm = NVMemoryManager(self, self.vram_size - (256 << 20), boot_size=(2 << 20), pt_t=NVPageTableEntry, va_bits=bits, va_shifts=shifts,
      va_base=0, palloc_ranges=[(x, x) for x in [512 << 20, 2 << 20, 4 << 10]], reserve_ptable=not self.large_bar)

  def _alloc_boot_mem(self, size:int, data:bytes|None=None, contiguous:bool=False, sysmem:bool|None=None) -> tuple[MMIOInterface,int|None,list[int]]:
    sz = round_up(size, 0x1000)
    if sysmem is True or (sysmem is None and not self.large_bar):
      view, sysaddr = self.pci_dev.alloc_sysmem(size, 0, contiguous=contiguous)
      paddr = None
    else:
      paddr = self.mm.palloc(sz, boot=False)
      view = self.vram.view(paddr, sz)
      sysaddr = [self.pci_dev.bar_info(1)[0] + paddr + i * 0x1000 for i in range(sz // 0x1000)]
    if data is not None: view[:size] = data
    return view, paddr, sysaddr

  def include(self, name:str, arch:str):
    for k,v in getattr(getattr(tinygrad.runtime.autogen.nv_regs, name), arch or 'regs').items():
      self.__dict__[k] = NVReg(self, *v) if isinstance(v, tuple) else v
