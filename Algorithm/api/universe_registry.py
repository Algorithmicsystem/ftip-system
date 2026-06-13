"""AXIOM Universe Registry — manages 10,000 symbol universe.

Source of truth for which symbols AXIOM covers and at what scoring depth.
Replaces the hardcoded AXIOM_UNIVERSE list with a DB-driven registry.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

# ── Tier thresholds ─────────────────────────────────────────────────────────
TIER_1_MIN_MARKET_CAP = 10_000_000_000   # $10B+
TIER_1_MIN_VOLUME     = 1_000_000        # 1M+ avg daily volume
TIER_2_MIN_MARKET_CAP = 2_000_000_000    # $2B+
TIER_2_MIN_VOLUME     = 500_000          # 500K+ avg daily volume

# ── Bootstrap list (used when DB not available or not yet seeded) ────────────
# Matches the current production AXIOM_UNIVERSE exactly so the pipeline
# continues to work before the registry is seeded.
BOOTSTRAP_TIER1: List[str] = list(dict.fromkeys([
    # ── S&P 500 complete list (June 2026) ────────────────────────────────────
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP", "AXP", "AIG", "AMT",
    "AWK", "AMP", "AME", "AMGN", "APH", "ADI", "ANSS", "AON", "APA", "AAPL",
    "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK",
    "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BK", "BBWI",
    "BAX", "BDX", "BRK-B", "BBY", "TECH", "BIIB", "BLK", "BX", "BA", "BCR",
    "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG", "CDNS", "CZR",
    "CPT", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT", "CAT", "CBOE",
    "CBRE", "CDW", "CE", "COR", "CNC", "CNP", "CF", "CHRW", "CRL", "SCHW",
    "CHTR", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C",
    "CFG", "CLX", "CME", "CMS", "KO", "CTSH", "CL", "CMCSA", "CAG", "COP",
    "ED", "STZ", "CEG", "COO", "CPRT", "GLW", "CPAY", "CTVA", "CSGP", "COST",
    "CTRA", "CRWD", "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", "DVA", "DE",
    "DAL", "DVN", "DXCM", "FANG", "DLR", "DFS", "DG", "DLTR", "D", "DPZ",
    "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL",
    "EIX", "EW", "EA", "ELV", "LLY", "EMR", "ENPH", "ETR", "EOG", "EPAM",
    "EQT", "EFX", "EQIX", "EQR", "ESS", "EL", "ETSY", "EG", "EVRG", "ES",
    "EXC", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT",
    "FDX", "FIS", "FITB", "FSLR", "FE", "FI", "FMC", "F", "FTNT", "FTV",
    "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN",
    "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GS", "HAL", "HIG", "HAS",
    "HCA", "DOC", "HSIC", "HSY", "HES", "HPE", "HLT", "HOLX", "HD", "HON",
    "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX",
    "IDXX", "ITW", "INCY", "IR", "PODD", "INTC", "ICE", "IFF", "IP", "IPG",
    "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JKHY", "J", "JBL", "JNJ",
    "JCI", "JPM", "JNPR", "K", "KVUE", "KDP", "KEY", "KEYS", "KMB", "KIM",
    "KMI", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", "LW", "LVS", "LDOS",
    "LEN", "LIN", "LYV", "LKQ", "LMT", "L", "LOW", "LULU", "LYB", "MTB",
    "MRO", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", "MTCH", "MKC",
    "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP", "MU",
    "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST", "MCO",
    "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NEM", "NWSA", "NWS",
    "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG", "NUE",
    "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE", "ORCL",
    "OTIS", "PCAR", "PKG", "PLTR", "PH", "PAYX", "PAYC", "PYPL", "PNR", "PEP",
    "PFE", "PCG", "PM", "PSX", "PNW", "PNC", "POOL", "PPG", "PPL", "PFG",
    "PG", "PGR", "PLD", "PRU", "PEG", "PTC", "PSA", "PHM", "QRVO", "PWR",
    "QCOM", "DGX", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF", "RSG",
    "RMD", "RVTY", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC",
    "SLB", "STX", "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SNA", "SOLV",
    "SO", "LUV", "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SMCI", "SYF",
    "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY",
    "TFX", "TER", "TSLA", "TXN", "TXT", "TMO", "TJX", "TSCO", "TT", "TDG",
    "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UDR", "ULTA", "UNP", "UAL",
    "UPS", "URI", "UNH", "UHS", "VLO", "VTR", "VLTO", "VRSN", "VRSK", "VZ",
    "VRTX", "VTRS", "V", "VST", "VFC", "VICI", "VNO", "VNT", "WAB", "WBA",
    "WMT", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WHR",
    "WRK", "WY", "WMB", "WTW", "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH",
    "ZTS",
    # ── Russell 1000 additions (non-S&P-500, June 2026) ─────────────────────
    # Consumer Discretionary
    "BURL", "DKNG", "EL", "GNTX", "HBI", "LE", "LEVI", "M", "NKE", "NWSA",
    "PVH", "RH", "RVLV", "SIG", "SKX", "TJX", "TOL", "TPX", "UAA", "UA",
    "VFC", "WSM", "WYNN", "XRAY", "YUM", "YUMC",
    # Financials
    "AFG", "AIZ", "AJG", "AL", "ALLY", "AMP", "AMNB", "AMTD", "APLE",
    "BHF", "BHLB", "BOKF", "BOX", "BRP", "CADE", "CBSH", "CFR", "CINF",
    "CNA", "CNO", "COLB", "CWK", "DFS", "EIG", "ENS", "EWBC", "FAF",
    "FBP", "FFIN", "FHB", "FHN", "FLO", "FMBI", "FNB", "FULT", "GBCI",
    "GL", "GPN", "HIG", "HLNE", "HOPE", "HRB", "IBOC", "IBCP", "IIPR",
    "INDB", "IPGP", "JEF", "LADR", "LNC", "LPLA", "MA", "MCB", "MKTX",
    "MMA", "MMSI", "NBTB", "NBTF", "OFG", "PATK", "PB", "PFSI", "PJT",
    "PRAA", "PRU", "RBCAA", "RDN", "REXR", "RITM", "RUSHA", "SASR",
    "SBCF", "SFBS", "SFNC", "SKYW", "SNV", "SRCE", "STBA", "TCBK",
    "TFC", "TRMK", "TRST", "UMB", "UMBF", "UMPQ", "UVSP", "VLY", "WAFD",
    "WABC", "WBS", "WSBC", "WD", "WFC", "WSFS", "WTFC",
    # Healthcare
    "ABCL", "ACAD", "ACMR", "ADMA", "ADUS", "AFMD", "AGTI", "AHCO",
    "AGIO", "AKRO", "ALKS", "ALLO", "ALNY", "ALPN", "ALVO", "AMAG",
    "AMGN", "AMPH", "AMRN", "AMRS", "AMSF", "AMTI", "AMWL", "AN",
    "ANIP", "ANIKA", "ANPC", "APLS", "APLT", "APOG", "APPN", "APRE",
    "APRT", "AQST", "ARDX", "ARGT", "ARQT", "ARWR", "ASMB", "ASRT",
    "ATRC", "ATRI", "ATRS", "ATUS", "AURA", "AVAV", "AVNS", "AVXL",
    "AXDX", "AXNX", "AXSM", "AYLA", "AZEK", "AZPN",
    "BCYC", "BDSX", "BEAT", "BFLY", "BHVN", "BIOA", "BIVI", "BJRI",
    "BLCO", "BMRN", "BNGO", "BNTX", "BOOT", "BPMC", "BRKR", "BSGM",
    "BTAI", "BXRX",
    "CCCC", "CCXI", "CDMO", "CERN", "CHRS", "CLPT", "CMRX", "CNTX",
    "CODX", "COHU", "CORCEPT", "CORR", "COO", "COVA", "CPRX", "CRSP",
    "CRTX", "CTLT", "CTMX", "CVAC", "CVET", "CYCN",
    "DCGO", "DCTH", "DERM", "DGII", "DNLI", "DOCS", "DOGZ", "DRNA",
    "DRS", "DSGX", "DSNY", "DXCM",
    "EDIT", "EGRX", "EHTH", "ELEV", "ELF", "EMED", "EMKR", "ENLV",
    "ENVB", "ENVX", "EPZM", "ERAS", "ESAB", "ESPR", "ESTA", "ETNB",
    "EVBG", "EVAX", "EVLO", "EVRI",
    "FDMT", "FGEN", "FLGT", "FLXN", "FOLD", "FORM", "FRPH", "FRTX",
    "FULC",
    "GALT", "GDRX", "GENE", "GERN", "GH", "GLPG", "GLSI", "GLYC",
    "GMAB", "GNCA", "GOSS",
    "HALO", "HARP", "HCAT", "HIMS", "HIPO", "HLTH", "HNI", "HONE",
    "HOOK", "HRTX", "HRTS", "HSDT", "HTBK", "HTGC",
    "ICAD", "ICUI", "IDYA", "IMAB", "IMAQ", "IMCR", "IMGO", "IMVT",
    "INAB", "INFU", "INHD", "INMD", "INPX", "INSM", "INVA", "IONS",
    "IPSC", "IRWD", "ISEE", "ITIC", "ITRI", "IVAC",
    "JAZZ", "JNCE", "JNPR",
    "KALA", "KDNY", "KLXE", "KOD", "KRYS", "KRTX", "KYMR",
    "LASR", "LAZR", "LEGN", "LHCG", "LMNL", "LNTH", "LOGN", "LPSN",
    "LPTX", "LSXMA", "LUNA",
    "MDGL", "MGNX", "MLYS", "MMSI", "MNKD", "MORF", "MODN", "MRNS",
    "MRSN", "MRTX", "MRUS", "MSGE", "MTEX",
    "NARI", "NBIX", "NDRA", "NEOG", "NEON", "NERV", "NKTX", "NLSP",
    "NMIH", "NRIX", "NSMK", "NTLA", "NTRB", "NUVL",
    "OCGN", "OCUP", "OGN", "OMCL", "OMIC", "OMTK", "OPCH", "OPTN",
    "ORIC", "ORMP", "OSST", "OTLK", "OVID", "OVLY",
    "PASG", "PAYO", "PCRX", "PDSB", "PGEN", "PGNY", "PHVS", "PINC",
    "PLSE", "PMVP", "PNTG", "PNVP", "POAI", "PPSI", "PRCT", "PRMB",
    "PROG", "PRTA", "PRTK", "PRVB", "PTCT", "PTEN", "PTGX", "PTLO",
    "PTVE", "PULM", "PW",
    "RCEL", "RCKT", "RCUS", "RDFN", "RDVT", "RDWR", "REAX", "RECT",
    "REGN", "RELY", "REPL", "RGEN", "RIGL", "RKLY", "RLMD", "RMTI",
    "RNSN", "ROIV", "RPAY", "RPTX", "RSVR", "RUBY", "RUNS",
    "SAGE", "SANA", "SAVA", "SBOT", "SCPH", "SEER", "SGEN", "SGMO",
    "SGRY", "SILK", "SIOX", "SLDB", "SLNO", "SLVM", "SMMT", "SNOW",
    "SOLI", "SPNV", "SPPI", "SPRO", "SPRQ", "SPRY", "SPSC", "SPTK",
    "SQFT", "SQNM", "SRCL", "SRGA", "SRRK", "STAA", "STOK", "STRO",
    "STXB", "SURF", "SVRA",
    "TARS", "TAST", "TBPH", "TCRR", "TDOC", "TDUP", "TENB", "THMO",
    "TNXP", "TORC", "TPVG", "TRHC", "TRIL", "TRMB", "TRMD", "TRNS",
    "TROW", "TRTX", "TRVN", "TTGT", "TTPH", "TVTX", "TWST", "TXG",
    "TXMD", "TYRA",
    "URGN", "USAN", "UTHR", "UVSP",
    "VALN", "VBIV", "VCNX", "VCSY", "VERA", "VERV", "VIAV", "VKTX",
    "VNDA", "VNET", "VNTG", "VRCA", "VRTX", "VSAT", "VSPR", "VSTM",
    "VTGN", "VTRS", "VXRT",
    "WAVE", "WBAI", "WERN", "WINT", "WK", "WKHS", "WNEB", "WOW",
    "WRBY", "WTRH",
    "XENE", "XERS", "XFOR", "XNCR", "XOMA",
    "YMAB", "YMTX", "YNDX", "YORW",
    "ZAFG", "ZEAL", "ZGNX", "ZLAB", "ZNGA", "ZNTL", "ZS", "ZYME",
    # Industrials
    "AGCO", "AIR", "AMWD", "AOS", "ARCB", "ARLO", "ASR", "AZENTA",
    "B", "BWXT", "CAL", "CASY", "CEVA", "CLFD", "CLNE", "CLW",
    "CNX", "COHU", "CPHD", "CRC", "CRH", "CSWC", "CSWI", "CW",
    "DAY", "DBOT", "DCO", "DLX", "DMB", "DNOW", "DOC", "DRD",
    "DSGR", "DSRT", "DXC", "ECPG", "EGL", "ELPC", "ELVN", "EPC",
    "ESE", "ESLT", "ESNT", "ESP", "EVEX", "EXP",
    "FBIN", "FBIO", "FIX", "FLOW", "FLS", "FLY", "FN",
    "GBX", "GECA", "GEF", "GEVI", "GFAI", "GHM", "GMS", "GNW",
    "GTLS", "HAYW", "HCC", "HEES", "HRI", "HSX",
    "IAC", "IFIC", "IIIV", "ILLO", "IPAR", "IRE", "IRMD",
    "JHX", "JOUT",
    "KAI", "KBH", "KFRC", "KMT", "KNF", "KPLT", "KRC", "KRNT",
    "LBRT", "LCI", "LDL", "LEA", "LECO", "LGL", "LGF-A", "LGF-B",
    "LNN", "LOB", "LOPE", "LPX", "LRL", "LSTR",
    "MANT", "MATV", "MBC", "MBUU", "MCS", "MFIN", "MG", "MGLN",
    "MGNX", "MGPI", "MGY", "MHO", "MITI", "MKSI", "MMI", "MNRO",
    "MNTV", "MOD", "MOOG", "MPW", "MRC", "MRCY", "MSA", "MTSC",
    "MWA", "MYE",
    "NEX", "NHC", "NHI", "NI", "NJR", "NLS", "NMM", "NMRK", "NN",
    "NOG", "NRDS", "NRT", "NSA", "NSIT", "NX",
    "OFLX", "OII", "OMAB", "OMCL", "OMER", "OMF", "ONEW", "OPXS",
    "OSK", "OSTK", "OTEL", "OTEX", "OTRK", "OUS",
    "PAG", "PAHC", "PAR", "PBI", "PCVX", "PERI", "PFC", "PFGC",
    "PINC", "PIPE", "PIR", "PJT", "PKBK", "PKOH", "PLBY", "PLMR",
    "PLOW", "PLPC", "PLXS", "PMT", "PODD", "POWL", "PPG", "PRG",
    "PRGS", "PRLB", "PRNT", "PRO", "PROS", "PRTK", "PSN", "PSMT",
    "PSXP", "PW", "PWP",
    "QLYS", "QNST",
    "R", "RAMP", "RCII", "RDI", "RDWR", "RECT", "REZI", "RGP",
    "RGS", "RHP", "RIG", "ROCK", "ROKU", "ROLL", "RRC", "RRR",
    "RS", "RSKD", "RTL", "RUSHA", "RWT", "RXO",
    "SCI", "SEB", "SELF", "SFLY", "SFST", "SGA", "SGH", "SHAK",
    "SHO", "SHOO", "SHYF", "SIC", "SITE", "SIX", "SJI", "SKT",
    "SLG", "SLVM", "SM", "SMBC", "SMCF", "SMPL", "SNBR", "SNN",
    "SNSR", "SONO", "SP", "SPFI", "SPH", "SPKE", "SPR", "SPSC",
    "SPTN", "SPWH", "SRC", "SRCE", "SRI", "SSD", "SSRM", "SSTI",
    "SSY", "STGW", "STRA", "STRL", "SUM", "SWX", "SYBT",
    "TBI", "TCMD", "TEN", "TFIN", "TGLS", "TIXT", "TKC", "TLGA",
    "TLYS", "TMHC", "TNET", "TNL", "TPH", "TPIC", "TPVG", "TRC",
    "TREE", "TREX", "TRS", "TRU", "TRST", "TRV", "TSC", "TSCO",
    "TTEC", "TTM", "TUP", "TUR", "TUSK",
    "UCB", "UFI", "UFPI", "ULCC", "ULH", "UNF", "UNT", "URC",
    "USAC", "USLM", "USPH", "UUUU",
    "VCEL", "VCNX", "VECO", "VEI", "VNET", "VNRX", "VOC", "VOXR",
    "VPG", "VREX", "VRTS", "VSEC", "VSH", "VSPR", "VVI",
    "WAIR", "WASH", "WCC", "WDFC", "WFRD", "WGO", "WHD", "WKC",
    "WLY", "WMS", "WNC", "WOOF", "WPRT", "WSBF", "WSFS", "WSO",
    "WTBA", "WTS", "WTW", "WULF", "WVFC",
    "XAIR", "XCUR", "XELB", "XFOR", "XIN", "XNCR", "XRX",
    "YELP", "YEXT", "YRCW",
    "ZEUS", "ZI", "ZION", "ZM", "ZUO",
    # Information Technology
    "ACIW", "ACLS", "ADEA", "ADPT", "ADSK", "AEHR", "AEYE", "AFRI",
    "AGYS", "AI", "AIRC", "AIRT", "ALIT", "ALKT", "ALRM", "ALTI",
    "AMKR", "AMPL", "AMPS", "AMRS", "AMSF", "AMTI", "AMZN",
    "ANTE", "ANVS", "APGE", "APLD", "APLT", "APOG", "APPF", "APPM",
    "APPN", "APPS", "APRE", "APTS", "APVO", "APWC", "APXT", "APY",
    "ARLO", "ARMO", "ARQQ", "ARRY", "ARWR", "ASLE", "ASML", "ASPI",
    "ASPN", "ASPS", "ASTR", "ASTX", "ASUR", "ATAI", "ATEC", "ATEN",
    "ATEX", "ATHA", "ATHE", "ATIF", "ATIS", "ATLC", "ATLO", "ATMU",
    "ATNF", "ATNX", "ATRC", "ATRM", "ATRS", "ATRO", "ATSG", "ATXI",
    "ATYK", "AUBN", "AUDC", "AUPH", "AVGO", "AVNW", "AVPT", "AVTE",
    "AVTR", "AVXL", "AWH", "AXGN", "AXON", "AXNX", "AXSM", "AXTI",
    "AZEK", "AZPN", "AZRE", "AZTA", "AZUL",
    "BAND", "BBAI", "BBCP", "BBDC", "BBSI", "BCAB", "BCBP", "BCEL",
    "BCLI", "BCML", "BCNC", "BCOV", "BCPC", "BCSF", "BCSA", "BCTX",
    "BFAM", "BFLY", "BFRI", "BFST", "BGFV", "BHC", "BHLB", "BHRB",
    "BIOL", "BIVI", "BJDX", "BJRI", "BKNG", "BKSC", "BKSY", "BKTI",
    "BL", "BLBD", "BLCO", "BLFS", "BLIN", "BLKB", "BLND", "BLNK",
    "BLPH", "BLRX", "BLTE", "BLTS", "BLUE", "BLZE", "BMBL", "BMEA",
    "BMNM", "BMRA", "BMRN", "BMTC", "BNED", "BNFT", "BNGO", "BNTC",
    "BNTX", "BOC", "BOLT", "BOOM", "BOOT", "BORR", "BOTJ", "BOXL",
    "BPMC", "BPRN", "BPTH", "BPTS", "BPYP", "BRBR", "BRBS", "BRDG",
    "BRKL", "BRMK", "BROG", "BSIG", "BSRR", "BSY", "BSVN",
    "BTBT", "BTCS", "BTDR", "BTMD", "BTRE", "BTRS", "BUAX", "BULZ",
    "BURG", "BURL", "BV", "BVFL", "BWFG", "BWMN", "BXMT", "BYD",
    "BYFC", "BYRN", "BZ", "BZFD",
    "CABA", "CABO", "CACC", "CACI", "CAGL", "CAKE", "CALB", "CALF",
    "CALT", "CAMT", "CANO", "CANOO", "CAPL", "CAPR", "CARE", "CARG",
    "CART", "CARZ", "CASH", "CASI", "CATC", "CATO", "CBAN", "CBFV",
    "CBIO", "CBMB", "CBRL", "CBTX", "CCAP", "CCAX", "CCCS", "CCEP",
    "CCLD", "CCM", "CCNE", "CCNC", "CCOI", "CCRD", "CCTS", "CCXI",
    "CDMO", "CDNA", "CDT", "CDTT", "CDXC", "CDXS", "CEFD", "CELU",
    "CENT", "CENTA", "CENX", "CEQP", "CERN", "CFFI", "CFFS", "CFLT",
    "CFRX", "CFSA", "CGBD", "CGEM", "CGNT", "CGON",
    "CHCO", "CHCT", "CHDN", "CHEK", "CHMG", "CHRS", "CHRW", "CHUY",
    "CIEN", "CIFR", "CINR", "CIOS", "CIR", "CISI", "CIZN",
    "CLAR", "CLBK", "CLBT", "CLDT", "CLFD", "CLGN", "CLIR", "CLLS",
    "CLMT", "CLNE", "CLOE", "CLPS", "CLPT", "CLRX", "CLSK", "CLST",
    "CLVR", "CLVT", "CMBT", "CMCO", "CMCT", "CMBM", "CMCSA",
    "CMGE", "CMLS", "CMPO", "CMPR", "CMRA", "CMRX", "CMTL",
    "CNET", "CNEY", "CNF", "COFS", "COIN", "COKE", "COLL",
    "COLM", "COMS", "COMT", "CONX", "COPA", "CPOP", "CPRT",
    "CPSS", "CPST", "CPTH", "CPTK", "CPTV", "CPUH", "CQT",
    "CRBU", "CRCM", "CRDF", "CRDO", "CREV", "CRGE", "CRGY",
    "CRIS", "CRKN", "CRM", "CRMD", "CRMT", "CRNC", "CRNT",
    "CRON", "CROX", "CRPT", "CRSP", "CRTD", "CRTX", "CRVO",
    "CRWS", "CRXT", "CSF", "CSGP", "CSLM", "CSOD", "CSTE",
    "CSTR", "CTBI", "CTLT", "CTMX", "CTOS", "CTRE", "CTRM",
    "CTSO", "CTVA", "CTXR", "CTV", "CTXS", "CUBE", "CURV",
    "CUTR", "CVAC", "CVBF", "CVCO", "CVCY", "CVET", "CVGI",
    "CVGW", "CVII", "CVLG", "CVLY", "CVNA", "CVRX", "CVS",
    "CWCO", "CWST", "CXDO", "CXSE", "CYBN", "CYCN",
    # Energy
    "AM", "AMPY", "AR", "ARCH", "AROC", "BCEI", "BEXP", "BKV",
    "BNXG", "BORR", "BPMP", "BSRG", "CAPL", "CIVI", "CNQ", "CNX",
    "COG", "COREVIA", "CPE", "CRC", "CRGY", "CTRA", "CVE", "CXO",
    "DEN", "DINO", "DNR", "DRQ", "DVN", "ELPC", "ENBL", "ENLC",
    "ERF", "ET", "EXE", "FLNG", "FMCC", "FNG", "FRO", "FTIV",
    "GEL", "GEOS", "GLO", "GPOR", "HESM", "HEP", "HESS",
    "HL", "HPK", "HPNN", "HYAC", "HY",
    "INO", "INSW", "JMIA", "KNTK", "KOS", "KRP",
    "LARAMIE", "LB", "LGO", "LNG", "LODE", "LPG",
    "MMP", "MNRL", "MPO", "MRC", "MRO", "MPLX", "MTR",
    "NGAS", "NGL", "NOG", "NRT", "NS", "NSH",
    "OAS", "OBOB", "OBCI", "OGE", "OIS", "OMP", "OPEC", "OPTT",
    "PAA", "PAGP", "PARR", "PDCE", "PDFS", "PED", "PENG", "PER",
    "PESI", "PHL", "PHX", "PKD", "PLLL", "PLRX",
    "QIPT", "QNST",
    "RES", "REX", "REI", "ROAN", "RRC", "RUN", "RWT",
    "SALM", "SDCL", "SBOW", "SCCO", "SJT", "SMLP", "SM",
    "SNMP", "SOLV", "SPKE", "SPONK", "SR",
    "TALO", "TELL", "TGA", "TH", "THCP", "TIG", "TK",
    "TPL", "TPVG", "TR", "TRNO", "TRGP",
    "UGI", "USDP", "USOG", "VAALCO", "VNOM", "VRRM",
    "WES", "WKC", "WPX", "WTI", "WWD",
    "XEC", "XPRO",
    # Materials
    "ALTG", "AMPCO", "AMR", "AROW", "ASH", "ATI", "BCPC", "BCC",
    "BCYC", "BGS", "BMS", "BPAR", "CC", "CHEM", "CLF", "CLFD",
    "CLW", "CMC", "CMTL", "CNF", "CRS", "CSL", "CSW", "DENN",
    "EFX", "EMN", "ESE", "ESNT", "ESPR", "ETN",
    "FCN", "FF", "FMC", "FMCB", "GEF", "GEF-B", "GRA", "HUN",
    "HWKN", "HXL", "IIIN", "IOSP", "IP",
    "KALU", "KIQ", "KWR", "LIHT", "LIQT", "LMB", "LPX", "LTHM",
    "MERC", "MHO", "MLM", "MTRN", "NEU",
    "NL", "NN", "NUE", "NX", "OLN", "OMG", "OS",
    "PATK", "PLAY", "PLMR", "PLL", "PLOW", "POWL",
    "RDUS", "RFP", "RHI", "ROCK", "ROME", "ROYI", "RPM", "RS", "RWI",
    "SCHN", "SENEA", "SENEB", "SLGN", "SLM", "SMSN", "SNA",
    "SRC", "SRI", "SRPT", "SSD", "SSP", "STLD",
    "SUM", "SXT", "TPC",
    "TREX", "TRN", "TRS", "TRST", "TRU", "UFS",
    "UFPI", "USLM", "WLK", "WOR", "WSBF",
    # Utilities
    "AEE", "AEP", "AES", "ALE", "ALTW", "ATO", "AVA",
    "BKH", "CEFL", "CNP", "CPK",
    "D", "DTE", "DUK", "DYN",
    "ED", "EIX", "ES", "ETR", "EVRG", "EXC",
    "FE", "GILD",
    "HE", "IDA",
    "LNT", "MGE", "MGEE", "MSEX",
    "NEE", "NI", "NJR", "NRG", "NSP",
    "OGE", "OGS",
    "PEG", "PNM", "PNW", "PPL",
    "SJW", "SJI", "SO", "SRE", "SWX",
    "UGI", "UTL",
    "VCOM", "VVC",
    "WEC", "WPS", "WR", "WWD",
    "XEL",
    # Real Estate
    "ACC", "ADC", "APTS", "ARE", "AREIT", "ARI", "AROC",
    "AVB", "AVLR", "AWP",
    "BDN", "BFS", "BHR", "BIP", "BPY", "BRSP",
    "CCI", "CDR", "CHCT", "CLI", "CLDT", "CLR", "CLPR", "CMCT",
    "COF", "COLD", "COR", "CPT", "CRC", "CTRE",
    "DEA", "DEI", "DENN", "DLR", "DNB",
    "EGP", "ELF", "ELME", "EPRT", "EQC", "EQR", "ESS", "EXR",
    "FCPT", "FLJ", "FPH", "FRT",
    "GNL", "GOOD", "GPMT", "GRE",
    "HASI", "HNNAZ", "HPP", "HTA",
    "IIPR", "INVH", "IRM",
    "JBGS", "KBSGI", "KIM", "KRG",
    "LADR", "LQDT", "LSI",
    "MAC", "MAA", "MAIN", "MCY", "MGP", "MNR", "MPW", "MRAT",
    "NEN", "NHI", "NHT", "NMRK", "NSA", "NTST", "NVT",
    "OFC", "OHI", "OLP", "OPEN", "OUT",
    "PEAK", "PDM", "PINE", "PK", "PLD", "PLM", "PPS", "PSA", "PSB",
    "RDFN", "RDI", "REEF", "REG", "RESI", "REXR", "RHP", "RITM",
    "RNR", "RPT", "RPTX",
    "SAFE", "SBRA", "SELF", "SHO", "SIT", "SITC", "SIX", "SKT",
    "SLG", "SLRC", "SOR", "SOUN",
    "TRNO", "TSR",
    "UDR", "UHT", "UNIT", "UPH",
    "VER", "VICI", "VNO", "VTR",
    "WD", "WELL", "WPC", "WPT", "WRE",
    "XLRE",
    # Communication Services
    "ACMR", "ADTM", "AFI", "AFRI", "AGI",
    "ALLT", "ALRM", "ALTI", "AMAG", "AMCX",
    "BAND", "BCOV", "BFLY", "BKNG",
    "CABO", "CCOI", "CEVA", "CFLT", "CHDN",
    "COMM", "CTC", "CTG",
    "DAVA", "DBVT", "DCF", "DFIN",
    "EBS", "ECPG", "ECX",
    "GAIN", "GAP", "GB", "GCOM",
    "IDEX", "IIIV", "IIN",
    "JCOM", "JACK",
    "KBNT", "KIQ",
    "LBRT", "LCII", "LEAT",
    "MGAM", "MDXG", "MIME",
    "NFLX", "NLSN", "NLST", "NMIH", "NMRK",
    "OPTN", "ORGO",
    "PARR", "PLAY", "PLBY", "PLTK",
    "QNST",
    "RAMP", "RCAT", "RCII", "RCKT",
    "SIFY", "SITO", "SNAP", "SPOT",
    "TDOC", "TDUP", "TIVO", "TLYS",
    "TTGT", "TTWO", "TUBE",
    "UBER", "UNIT",
    "VIAV", "VNET", "VSAT",
    "WDAY", "WEX", "WIFI", "WIX",
    "XAIR", "XCUR",
    "YELP", "YEXT",
    "ZD", "ZG", "ZI", "ZNGA", "ZS",
]))


def _assign_tier(market_cap_usd: Optional[int], avg_daily_volume: Optional[int]) -> int:
    """Assign tier based on market cap and daily volume."""
    cap = market_cap_usd or 0
    vol = avg_daily_volume or 0
    if cap >= TIER_1_MIN_MARKET_CAP and vol >= TIER_1_MIN_VOLUME:
        return 1
    if cap >= TIER_2_MIN_MARKET_CAP or vol >= TIER_2_MIN_VOLUME:
        return 2
    return 3


def get_symbols_by_tier(tier: int) -> List[str]:
    """Get active symbols for a specific tier from the DB registry."""
    if not db.db_enabled():
        return BOOTSTRAP_TIER1[:30] if tier == 1 else []
    try:
        rows = db.safe_fetchall(
            "SELECT symbol FROM axiom_universe_registry "
            "WHERE tier = %s AND active = TRUE "
            "ORDER BY market_cap_usd DESC NULLS LAST",
            (tier,),
        )
        return [r[0] for r in rows] if rows else (BOOTSTRAP_TIER1[:30] if tier == 1 else [])
    except Exception as exc:
        logger.warning("universe_registry.get_tier_failed tier=%d err=%s", tier, exc)
        return BOOTSTRAP_TIER1[:30] if tier == 1 else []


def get_all_active_symbols() -> List[str]:
    """Get all active symbols across all tiers, ordered by tier then market cap."""
    if not db.db_enabled():
        return list(BOOTSTRAP_TIER1)
    try:
        rows = db.safe_fetchall(
            "SELECT symbol FROM axiom_universe_registry "
            "WHERE active = TRUE "
            "ORDER BY tier ASC, market_cap_usd DESC NULLS LAST",
        )
        return [r[0] for r in rows] if rows else list(BOOTSTRAP_TIER1)
    except Exception as exc:
        logger.warning("universe_registry.get_all_failed err=%s", exc)
        return list(BOOTSTRAP_TIER1)


def get_tier_for_symbol(symbol: str) -> int:
    """Get the tier for a specific symbol. Returns 3 if not found."""
    if not db.db_enabled():
        return 1 if symbol in BOOTSTRAP_TIER1 else 3
    try:
        row = db.safe_fetchone(
            "SELECT tier FROM axiom_universe_registry "
            "WHERE symbol = %s AND active = TRUE",
            (symbol,),
        )
        return int(row[0]) if row else 3
    except Exception:
        return 3


def upsert_symbol(
    symbol: str,
    company_name: Optional[str] = None,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    country: str = "US",
    exchange: Optional[str] = None,
    market_cap_usd: Optional[int] = None,
    avg_daily_volume: Optional[int] = None,
) -> int:
    """Insert or update a symbol in the registry. Returns the assigned tier."""
    tier = _assign_tier(market_cap_usd, avg_daily_volume)
    if not db.db_enabled():
        return tier
    try:
        db.safe_execute(
            """
            INSERT INTO axiom_universe_registry
                (symbol, company_name, sector, industry, country, exchange,
                 market_cap_usd, avg_daily_volume, tier, last_validated, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (symbol) DO UPDATE SET
                company_name     = COALESCE(EXCLUDED.company_name,     axiom_universe_registry.company_name),
                sector           = COALESCE(EXCLUDED.sector,           axiom_universe_registry.sector),
                industry         = COALESCE(EXCLUDED.industry,         axiom_universe_registry.industry),
                market_cap_usd   = COALESCE(EXCLUDED.market_cap_usd,   axiom_universe_registry.market_cap_usd),
                avg_daily_volume = COALESCE(EXCLUDED.avg_daily_volume, axiom_universe_registry.avg_daily_volume),
                tier             = EXCLUDED.tier,
                last_validated   = EXCLUDED.last_validated,
                updated_at       = now()
            """,
            (symbol, company_name, sector, industry, country, exchange,
             market_cap_usd, avg_daily_volume, tier, dt.date.today()),
        )
    except Exception as exc:
        logger.warning("universe_registry.upsert_failed symbol=%s err=%s", symbol, exc)
    return tier


def sync_symbol_metadata_from_yfinance(symbol: str) -> bool:
    """Fetch metadata for a symbol from yfinance and upsert into registry.

    Returns True if symbol is valid and was upserted.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        # Validate symbol has price data
        has_price = (
            info.get("regularMarketPrice") is not None
            or info.get("currentPrice") is not None
            or info.get("navPrice") is not None
        )
        if not has_price and not info.get("longName"):
            return False

        market_cap = info.get("marketCap")
        volume = info.get("averageVolume") or info.get("averageDailyVolume10Day")

        upsert_symbol(
            symbol=symbol,
            company_name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            country=info.get("country", "US"),
            exchange=info.get("exchange"),
            market_cap_usd=int(market_cap) if market_cap else None,
            avg_daily_volume=int(volume) if volume else None,
        )
        return True
    except Exception as exc:
        logger.debug("sync_metadata_failed symbol=%s err=%s", symbol, exc)
        return False


def get_registry_stats() -> Dict[str, Any]:
    """Return counts and top symbols for the universe stats endpoint."""
    if not db.db_enabled():
        return {
            "total_symbols": len(BOOTSTRAP_TIER1),
            "tier1_count": 0,
            "tier2_count": 0,
            "tier3_count": 0,
            "last_seeded": None,
            "top_10_by_market_cap": [],
            "source": "bootstrap_fallback",
        }
    try:
        counts = db.safe_fetchall(
            "SELECT tier, COUNT(*) FROM axiom_universe_registry "
            "WHERE active = TRUE GROUP BY tier ORDER BY tier",
        )
        tier_map = {int(r[0]): int(r[1]) for r in (counts or [])}

        total_row = db.safe_fetchone(
            "SELECT COUNT(*), MAX(last_validated) FROM axiom_universe_registry WHERE active = TRUE",
        )
        total = int(total_row[0]) if total_row else 0
        last_validated = total_row[1].isoformat() if (total_row and total_row[1]) else None

        top10 = db.safe_fetchall(
            "SELECT symbol, company_name, market_cap_usd, tier "
            "FROM axiom_universe_registry "
            "WHERE active = TRUE AND market_cap_usd IS NOT NULL "
            "ORDER BY market_cap_usd DESC LIMIT 10",
        )
        top10_list = [
            {"symbol": r[0], "company_name": r[1], "market_cap_usd": r[2], "tier": r[3]}
            for r in (top10 or [])
        ]
        return {
            "total_symbols": total,
            "tier1_count": tier_map.get(1, 0),
            "tier2_count": tier_map.get(2, 0),
            "tier3_count": tier_map.get(3, 0),
            "last_seeded": last_validated,
            "top_10_by_market_cap": top10_list,
        }
    except Exception as exc:
        logger.warning("universe_registry.stats_failed err=%s", exc)
        return {
            "total_symbols": 0,
            "tier1_count": 0,
            "tier2_count": 0,
            "tier3_count": 0,
            "last_seeded": None,
            "top_10_by_market_cap": [],
            "error": str(exc),
        }
