##################################
# priv/extensions/ZicntrCommon.py
#
# Shared Zicntr test generation for the Sm/S/U counter-enable suites.
# David_Harris@hmc.edu 30 August 2026
# SPDX-License-Identifier: Apache-2.0
##################################

"""Functions for generating Zicntr counter-enable tests in all priv modes"""

from typing import Literal

from testgen.asm.helpers import comment_banner, write_sigupd
from testgen.asm.tsbi import tsbi_call
from testgen.data.state import TestData
from testgen.priv.extensions.InterruptsCommon import csr_access

Mode = Literal["M", "S", "U"]
Counteren = Literal["ones", "zeros"]

_COUNTERS = ["cycle", "time", "instret"]

# Instructions retired by one trip round the wfi wait loop in instret_interrupt_tests:
# LREG, bne, wfi, addi, j. Keep this equal to the loop body; the wait is subtracted using it.
_WAIT_LOOP_INSNS = 5


def _access_counter(
    test_data: TestData, covergroup: str, coverpoint: str, bin_prefix: str, read_reg: int, i: int
) -> list[str]:
    """Read counter i and attempt to write it, low half and high half on RV32.

    The read traps or not according to mcounteren and scounteren; the write always raises an illegal
    instruction, because the unprivileged counters are read-only CSRs.
    """

    def access(name: str, suffix: str) -> list[str]:
        return [
            test_data.add_testcase(f"{bin_prefix}_read{suffix}", coverpoint, covergroup),
            f"csrr x{read_reg}, {name}",
            test_data.add_testcase(f"{bin_prefix}_write{suffix}", coverpoint, covergroup),
            f"csrw {name}, zero  # read-only CSR: illegal instruction whatever the counterens hold",
        ]

    if i < 3:
        name = _COUNTERS[i]
        return [
            *access(name, ""),
            "#if __riscv_xlen == 32",
            *access(f"{name}h", "h"),
            "#endif",
        ]
    # Access only the hpmcounters the configuration implements: an unimplemented counter may trap or
    # return a constant (norm:hpm_unimplemented_counter_access), so no reference signature fits both.
    # UDB_HPM_COUNTER_EN_<n> comes from the HPM_COUNTER_EN parameter of the UDB config.
    return [
        f"#if defined(ZIHPM_SUPPORTED) && defined(UDB_HPM_COUNTER_EN_{i})",
        *access(f"hpmcounter{i}", ""),
        "#if __riscv_xlen == 32",
        *access(f"hpmcounter{i}h", "h"),
        "#endif",
        "#endif",
    ]


def _write_counteren(csr: str, operand: str, mode: Mode, comment: str = "") -> str:
    """Write csr directly when mode can, otherwise through T-SBI: mcounteren is M-mode only,
    scounteren is writable from M and S."""
    instr = f"csrw {csr}, {operand}"
    if comment:
        instr += f"  # {comment}"
    if mode == "M" or (mode == "S" and csr == "scounteren"):
        return instr
    return tsbi_call(instr)


def counteren_walk_tests(
    test_data: TestData,
    covergroup: str,
    coverpoint: str,
    description: str,
    *,
    csrs: list[str],
    mode: Mode,
    mcounteren: Counteren | None = None,
    scounteren_ones: bool = False,
    tag: str = "",
) -> list[str]:
    """
    Walk a 1 and then a 0 through every bit of each CSR in csrs (the same value in each), reading
    every counter after each write. Everything runs in mode; writes that mode cannot make directly
    go through T-SBI. mcounteren optionally presets that register to all ones or all zeros first, and
    scounteren_ones presets scounteren to all ones when S-mode exists.
    tag prefixes the testcase names so a coverpoint tested with several mcounteren settings stays unique.
    """
    read_reg, ones_reg, walk_reg, inv_reg = test_data.int_regs.get_registers(4)
    lines = [comment_banner(coverpoint, description), ""]
    if scounteren_ones:
        lines += [
            "#ifdef S_SUPPORTED",
            f"LI(x{ones_reg}, -1)",
            _write_counteren("scounteren", f"x{ones_reg}", mode, "enable all counters for U-mode"),
            "#endif // S_SUPPORTED",
        ]
    if mcounteren == "ones":
        lines += [
            f"LI(x{ones_reg}, -1)",
            _write_counteren("mcounteren", f"x{ones_reg}", mode, "enable all counters"),
        ]
    elif mcounteren == "zeros":
        lines.append(_write_counteren("mcounteren", "zero", mode, "disable all counters"))

    lines.append(f"LI(x{walk_reg}, 1)")
    for i in range(32):
        lines += [
            *(_write_counteren(csr, f"x{walk_reg}", mode, "set only the current bit") for csr in csrs),
            *_access_counter(test_data, covergroup, coverpoint, f"{tag}walking_1_{i}", read_reg, i),
            f"slli x{walk_reg}, x{walk_reg}, 1",
        ]

    lines.append(f"LI(x{walk_reg}, 1)")
    for i in range(32):
        lines += [
            f"not x{inv_reg}, x{walk_reg}  # all bits but the current one",
            *(_write_counteren(csr, f"x{inv_reg}", mode, "clear only the current bit") for csr in csrs),
            *_access_counter(test_data, covergroup, coverpoint, f"{tag}walking_0_{i}", read_reg, i),
            f"slli x{walk_reg}, x{walk_reg}, 1",
        ]
    test_data.int_regs.return_registers([read_reg, ones_reg, walk_reg, inv_reg])
    return lines


def _set_counterens(operand: str, mode: Mode) -> list[str]:
    """Write mcounteren, plus scounteren when it also gates the running mode."""
    lines = [_write_counteren("mcounteren", operand, mode)]
    if mode == "U":
        lines += ["#ifdef S_SUPPORTED", _write_counteren("scounteren", operand, mode), "#endif"]
    return lines


def counter_inc_inaccessible_tests(test_data: TestData, covergroup: str, mode: Mode) -> list[str]:
    """Check that instret keeps counting while it is inaccessible in mode."""
    coverpoint = "cp_mcounter_inc_inaccessible"
    description = (
        f"running in {mode} mode\n"
        "enable counters and read instret\n"
        f"disable counters so instret is inaccessible in {mode} mode\n"
        "re-enable counters; the instructions doing so retire while instret is inaccessible\n"
        "read and sigupd change in instret"
    )

    old_reg, read_reg = test_data.int_regs.get_registers(2)

    lines = [
        comment_banner(coverpoint, description),
        "",
        test_data.add_testcase(mode, coverpoint, covergroup),
        f"# make counter accessible in {mode} mode",
        f"LI(x{read_reg}, -1)",
        *_set_counterens(f"x{read_reg}", mode),
        f"csrr x{old_reg}, instret",
        f"# make counter inaccessible in {mode} mode",
        *_set_counterens("zero", mode),
        f"# make counter accessible in {mode} mode",
        *_set_counterens(f"x{read_reg}", mode),
        f"csrr x{read_reg}, instret",
        f"sub x{read_reg}, x{read_reg}, x{old_reg}",
        "# SIGUPD the difference in instret",
        write_sigupd(read_reg, test_data),
    ]
    test_data.int_regs.return_registers([old_reg, read_reg])
    return lines


def _alloc(test_data: TestData, count: int) -> list[int]:
    regs = [test_data.int_regs.get_register()] if count == 1 else list(test_data.int_regs.get_registers(count))
    assert 10 not in regs, "x10 (a0) is clobbered by T-SBI and ecall; do not keep a live value in it"
    return regs


def _instret_case(
    test_data: TestData,
    covergroup: str,
    name: str,
    mode: Mode,
    body: list[str],
    *,
    setup: list[str] | None = None,
    cleanup: list[str] | None = None,
) -> list[str]:
    before, after, diff = _alloc(test_data, 3)
    counter = "instret" if mode == "U" else "minstret"
    lines = [
        *(setup or []),
        test_data.add_testcase(f"{counter}_{name}", "cp_instret_delta", covergroup),
        csr_access(f"csrr x{before}, {counter}", mode),
        *body,
        csr_access(f"csrr x{after}, {counter}", mode),
        f"sub x{diff}, x{after}, x{before}",
        write_sigupd(diff, test_data),
        *(cleanup or []),
        "",
    ]
    test_data.int_regs.return_registers([before, after, diff])
    return lines


def instret_retire_tests(test_data: TestData, covergroup: str, mode: Mode) -> list[str]:
    """Normally retiring instructions.

    M (ZicntrSm): add, mret, and sret when S exists. S (ZicntrS): sret. U (ZicntrU): add.
    """
    csr = "instret" if mode == "U" else "minstret"
    lines = [csr_access("csrw mcountinhibit, zero  # run all counters", mode)]
    if mode == "U":
        (prep,) = _alloc(test_data, 1)
        lines += [f"LI(x{prep}, -1)", *_set_counterens(f"x{prep}", mode)]
        test_data.int_regs.return_registers([prep])

    if mode in ("M", "U"):
        (tmp,) = _alloc(test_data, 1)
        lines += [
            comment_banner("cp_instret_delta", f"{csr} delta around a normally retiring add in {mode}-mode"),
            "",
            *_instret_case(test_data, covergroup, "add", mode, [f"add x{tmp}, zero, zero  # instruction under test"]),
        ]
        test_data.int_regs.return_registers([tmp])

    if mode == "M":
        save, mask = _alloc(test_data, 2)
        lines += [
            comment_banner(
                "cp_instret_delta", "minstret delta around mret (MPP forced to M, returns to the next line)"
            ),
            "",
            *_instret_case(
                test_data,
                covergroup,
                "mret",
                mode,
                ["mret  # instruction under test", "1:"],
                setup=[
                    f"csrr x{save}, mstatus  # save mstatus",
                    f"LI(x{mask}, 0x1800)  # MPP = M",
                    f"or x{mask}, x{mask}, x{save}",
                    f"csrw mstatus, x{mask}",
                    f"LA(x{mask}, 1f)  # return address after mret",
                    f"csrw mepc, x{mask}",
                ],
                cleanup=[f"csrw mstatus, x{save}  # restore mstatus"],
            ),
        ]
        test_data.int_regs.return_registers([save, mask])

        # sret returns to U-mode; T-SBI returns to M before reading minstret.
        save, tmp = _alloc(test_data, 2)
        lines += [
            "#ifdef S_SUPPORTED",
            comment_banner("cp_instret_delta", "minstret delta around sret from M-mode (SPP = U, T-SBI back to M)"),
            "",
            *_instret_case(
                test_data,
                covergroup,
                "sret",
                mode,
                [
                    "sret  # instruction under test",
                    "1:",
                    "RVTEST_TSBI_GOTO_MMODE  # back to M-mode before reading minstret",
                ],
                setup=[
                    f"csrr x{save}, sstatus  # save sstatus",
                    f"LI(x{tmp}, 0x100)",
                    f"csrc sstatus, x{tmp}  # SPP = 0",
                    f"LA(x{tmp}, 1f)  # return address after sret",
                    f"csrw sepc, x{tmp}",
                ],
                cleanup=[f"csrw sstatus, x{save}  # restore sstatus"],
            ),
            "#endif // S_SUPPORTED",
        ]
        test_data.int_regs.return_registers([save, tmp])

    if mode == "S":
        # sret returns to S-mode; minstret is read through T-SBI.
        save, tmp = _alloc(test_data, 2)
        lines += [
            comment_banner("cp_instret_delta", "minstret delta around sret in S-mode (SPP = S), read via T-SBI"),
            "",
            *_instret_case(
                test_data,
                covergroup,
                "sret",
                mode,
                ["sret  # instruction under test", "1:"],
                setup=[
                    f"csrr x{save}, sstatus  # save sstatus",
                    f"LI(x{tmp}, 0x400000)",
                    csr_access(f"csrc mstatus, x{tmp}  # mstatus.TSR = 0 so sret does not trap", mode),
                    f"LA(x{tmp}, 1f)  # return address after sret",
                    f"csrw sepc, x{tmp}",
                    f"LI(x{tmp}, 0x100)",
                    f"csrs sstatus, x{tmp}  # SPP = 1: sret stays in S-mode",
                ],
                cleanup=[f"csrw sstatus, x{save}  # restore sstatus"],
            ),
        ]
        test_data.int_regs.return_registers([save, tmp])

    return lines


def instret_exception_tests(test_data: TestData, covergroup: str, mode: Mode) -> list[str]:
    """Instructions that trap before retiring: ecall, ebreak, illegal, load access fault, load misaligned.

    The trapping instruction does not retire but the trap handler's instructions do, so the raw delta is
    recorded. Runs in M-mode (minstret) or U-mode (instret); ecall goes through T-SBI in both.
    """
    csr = "instret" if mode == "U" else "minstret"
    lines = [csr_access("csrw mcountinhibit, zero  # run all counters", mode)]
    if mode == "U":
        (prep,) = _alloc(test_data, 1)
        lines += [f"LI(x{prep}, -1)", *_set_counterens(f"x{prep}", mode)]
        test_data.int_regs.return_registers([prep])

    lines += [
        comment_banner("cp_instret_delta", f"ecall in {mode}-mode: traps before retiring, {csr} delta recorded"),
        "",
        *_instret_case(
            test_data,
            covergroup,
            "ecall",
            mode,
            ["RVTEST_TSBI_ECALL_TEST  # traps, resumes right after this line"],
        ),
        comment_banner("cp_instret_delta", f"ebreak in {mode}-mode: traps before retiring, {csr} delta recorded"),
        "",
        *_instret_case(test_data, covergroup, "ebreak", mode, ["ebreak", "nop"]),
        comment_banner(
            "cp_instret_delta", f"Illegal instruction in {mode}-mode: traps before retiring, {csr} delta recorded"
        ),
        "",
        *_instret_case(
            test_data,
            covergroup,
            "illegal",
            mode,
            [".word 0xFFFFFFFF", "nop"],
            setup=[".p2align 2"],
        ),
    ]

    addr, tmp = _alloc(test_data, 2)
    lines.append("#ifdef RVMODEL_ACCESS_FAULT_ADDRESS")
    lines += [
        comment_banner(
            "cp_instret_delta", f"Load access fault in {mode}-mode: traps before retiring, {csr} delta recorded"
        ),
        "",
        *_instret_case(
            test_data,
            covergroup,
            "load_access_fault",
            mode,
            [f"lw x{tmp}, 0(x{addr})"],
            setup=[f"LA(x{addr}, RVMODEL_ACCESS_FAULT_ADDRESS)"],
        ),
    ]
    lines += ["#endif // RVMODEL_ACCESS_FAULT_ADDRESS", ""]
    lines += [
        comment_banner(
            "cp_instret_delta", f"Load address misaligned in {mode}-mode: traps before retiring, {csr} delta recorded"
        ),
        "",
        *_instret_case(
            test_data,
            covergroup,
            "load_misaligned",
            mode,
            [f"lw x{tmp}, 0(x{addr})"],
            setup=[f"LA(x{addr}, scratch)", f"addi x{addr}, x{addr}, 1  # misalign by 1 byte"],
        ),
    ]
    test_data.int_regs.return_registers([addr, tmp])
    return lines


def instret_interrupt_tests(test_data: TestData, covergroup: str, mode: Mode) -> list[str]:
    """wfi and wrs interrupt cases."""
    csr = "instret" if mode == "U" else "minstret"
    soon = f"RVTEST_SET_MTIME_INT_SOON_{mode}"
    clr = f"RVTEST_CLR_MTIME_INT_{mode}"
    lines = [csr_access("csrw mcountinhibit, zero  # run all counters", mode)]
    if mode == "U":
        (prep,) = _alloc(test_data, 1)
        lines += [f"LI(x{prep}, -1)", *_set_counterens(f"x{prep}", mode)]
        test_data.int_regs.return_registers([prep])
    # The register pool is small: the three held here plus the three that _instret_case takes is the most it
    # can hand out at once. wfi_taken allocates its own three (so it can use them as loop scratch and tally),
    # and res is only allocated for the wrs cases, after addr has been returned.
    tmp, scr, addr = _alloc(test_data, 3)

    cond = "defined(UDB_WFI_FINITE)" + (" && defined(UDB_WFI_U_MODE)" if mode == "U" else "")
    lines += [
        f"#if {cond}",
        comment_banner("cp_instret_delta", f"wfi in {mode}-mode with nothing armed: finite wait, {csr} delta recorded"),
        "",
        *_instret_case(
            test_data,
            covergroup,
            "wfi_timeout",
            mode,
            ["wfi  # no event armed; falls through or times out", *(["nop"] if mode == "U" else [])],
            setup=[
                csr_access("csrw mie, zero  # nothing enabled", mode),
                *(["csrci mstatus, 8  # MIE = 0"] if mode == "M" else []),
                f"{clr}  # make sure nothing is pending",
            ],
        ),
        f"#endif // {cond}",
        "",
    ]

    if mode == "M":
        lines += [
            comment_banner("cp_instret_delta", "wfi with the timer interrupt pending and MIE = 0: retires, no trap"),
            "",
            *_instret_case(
                test_data,
                covergroup,
                "wfi_pending",
                mode,
                ["wfi  # interrupt already pending, MIE = 0 so it is not taken"],
                setup=[
                    csr_access("csrw mie, zero  # nothing enabled", mode),
                    "csrci mstatus, 8  # MIE = 0",
                    f"LI(x{tmp}, 0x80)",
                    csr_access(f"csrw mie, x{tmp}  # MTIE only", mode),
                    soon,
                    f"RVTEST_IDLE_FOR_INTERRUPT(x{tmp})  # wait for MTIP to become pending",
                ],
                cleanup=[clr],
            ),
        ]

    if mode == "U":
        lines.append("#ifdef UDB_WFI_U_MODE")
    # How long wfi waits is up to the DUT: it may sleep until the timer fires or return at once, so the number
    # of times the loop runs differs between DUTs. The loop repeats wfi until the trap count shows the interrupt
    # was taken, and every trip retires exactly _WAIT_LOOP_INSNS instructions, tallied in `diff`. Subtracting
    # that tally from the delta leaves only the part that does not depend on timing. Written out here rather
    # than through _instret_case because the loop borrows its registers: `after` is scratch until it is read,
    # and `diff` holds the tally until the delta is computed.
    before, after, diff = _alloc(test_data, 3)
    lines += [
        comment_banner(
            "cp_instret_delta", f"wfi in {mode}-mode: timer interrupt taken during the wait, {csr} delta recorded"
        ),
        "",
        csr_access("csrw mie, zero  # nothing enabled", mode),
        *(["csrci mstatus, 8  # MIE = 0"] if mode == "M" else []),
        f"LI(x{after}, 0x80)",
        csr_access(f"csrw mie, x{after}  # MTIE only", mode),
        soon,
        # After the last trap of the setup (T-SBI calls count as traps), so that only the timer
        # interrupt can change the count from here on.
        f"LA(x{addr}, rvtest_trap_count)",
        f"LREG x{scr}, 0(x{addr})  # trap count before waiting",
        f"LI(x{diff}, 0)  # tally of instructions retired by the wait loop",
        *(["csrsi mstatus, 8  # MIE = 1"] if mode == "M" else []),
        test_data.add_testcase(f"{csr}_wfi_taken", "cp_instret_delta", covergroup),
        csr_access(f"csrr x{before}, {csr}", mode),
        "1:",
        f"LREG x{after}, 0(x{addr})  # trap count now",
        f"bne x{after}, x{scr}, 2f  # timer interrupt has been taken: leave the loop",
        "wfi  # may sleep until the interrupt, or return early",
        f"addi x{diff}, x{diff}, {_WAIT_LOOP_INSNS}  # tally what this trip retired",
        "j 1b",
        "2:",
        csr_access(f"csrr x{after}, {csr}", mode),
        f"sub x{after}, x{after}, x{diff}  # remove the timing-dependent wait",
        f"sub x{diff}, x{after}, x{before}",
        write_sigupd(diff, test_data),
        *(["csrci mstatus, 8  # MIE = 0"] if mode == "M" else []),
        clr,
        "",
    ]
    test_data.int_regs.return_registers([before, after, diff, addr])
    (res,) = _alloc(test_data, 1)  # lr.w destination for the wrs cases
    if mode == "U":
        lines.append("#endif // UDB_WFI_U_MODE")

    lines += ["#ifdef ZAWRS_SUPPORTED", ""]
    lines += [
        comment_banner("cp_instret_delta", f"wrs.nto in {mode}-mode: {csr} delta recorded"),
        "",
        *_instret_case(
            test_data,
            covergroup,
            "wrs_nto",
            mode,
            [
                "#ifndef UDB_ZAWRS_NTO_IS_NOP",
                *(["csrsi mstatus, 8  # MIE = 1"] if mode == "M" else []),
                "#endif // UDB_ZAWRS_NTO_IS_NOP",
                "wrs.nto",
            ],
            setup=[
                csr_access("csrw mie, zero  # nothing enabled", mode),
                *(["csrci mstatus, 8  # MIE = 0"] if mode == "M" else []),
                f"LA(x{scr}, scratch)",
                f"lr.w x{res}, (x{scr})  # reservation for wrs",
                "#ifndef UDB_ZAWRS_NTO_IS_NOP",
                f"LI(x{tmp}, 0x80)",
                csr_access(f"csrw mie, x{tmp}  # MTIE only", mode),
                *(["csrsi mstatus, 8  # MIE = 1"] if mode == "M" else []),
                soon,
                "#endif // UDB_ZAWRS_NTO_IS_NOP",
            ],
            cleanup=[
                *(["csrci mstatus, 8  # MIE = 0"] if mode == "M" else []),
                "#ifndef UDB_ZAWRS_NTO_IS_NOP",
                clr,
                "#endif // UDB_ZAWRS_NTO_IS_NOP",
            ],
        ),
    ]

    lines += [
        comment_banner("cp_instret_delta", f"wrs.sto in {mode}-mode: timer interrupt taken, {csr} delta recorded"),
        "",
        *_instret_case(
            test_data,
            covergroup,
            "wrs_sto",
            mode,
            ["wrs.sto  # interrupt taken here"],
            setup=[
                csr_access("csrw mie, zero  # nothing enabled", mode),
                *(["csrci mstatus, 8  # MIE = 0"] if mode == "M" else []),
                f"LA(x{scr}, scratch)",
                f"lr.w x{res}, (x{scr})  # reservation for wrs",
                f"LI(x{tmp}, 0x80)",
                csr_access(f"csrw mie, x{tmp}  # MTIE only", mode),
                *(["csrsi mstatus, 8  # MIE = 1"] if mode == "M" else []),
                soon,
            ],
            cleanup=[*(["csrci mstatus, 8  # MIE = 0"] if mode == "M" else []), clr],
        ),
    ]
    lines.append("#endif // ZAWRS_SUPPORTED")

    test_data.int_regs.return_registers([tmp, res, scr])
    return lines
