//! benchmarks/msmarco_v1.zig — MS MARCO-v1 (dev.small) harness.
//!
//! Owner: retriever. Target operating points (paper Table 1):
//!   - MRR@10 = 39.0 → 10 ms/query
//!   - MRR@10 = 39.3 → 14 ms/query
//!
//! Single-thread (paper §9). Sweeps κ_c × κ_d × α over the paper's grid and
//! writes one CSV row per cell.
//!
//! Wiring: lead adds an executable step in build.zig once #20 unblocks:
//!     const bench_msmarco = b.addExecutable(.{
//!         .name = "bench_msmarco",
//!         .root_source_file = b.path("benchmarks/msmarco_v1.zig"),
//!         .target = target, .optimize = .ReleaseFast,
//!     });
//!     bench_msmarco.root_module.addImport("tac", lib_mod);
//!
//! ---------------------------------------------------------------------------
//! DESIGN PSEUDOCODE (waits on #15 Index.build, #19 refine, #5 encoder).
//! ---------------------------------------------------------------------------
//!
//! main():
//!   args = parseArgs() — index path, queries.bin, qrels.tsv, results.csv, [grid override]
//!   tac.eval.latency.pinToCore(0);
//!
//!   index = tac.index.storage.open(args.index_path);
//!   pq    = index.pq;        // PQ codebooks live inside the index handle
//!   queries = common.queries.load(args.queries_bin);
//!   qrels   = common.qrels.loadMsmarco(args.qrels_tsv);   // dict<qid, []doc_id>
//!
//!   csv = openCsv(args.results_csv);
//!   csv.writeHeader();
//!
//!   for kappa_c in [15, 20, 40, 80, 100, 120]:
//!     for kappa_d in [250, 500, 1000, 2000, 4000]:
//!       for alpha in [null, 0.35, 0.4, 0.45, 0.5]:
//!         params = SearchParams{ kappa_c, kappa_d, alpha, top_k: 10 };
//!         timings = ArrayList(StageTimings)
//!         mrr_sum = 0.0
//!         for q in queries:
//!           t0 = Timer.start()
//!           result = retrieval.search(index, pq, q.tokens, q.n_q, params, gpa)
//!           total = t0.read()
//!           // Per-stage timings come from a debug variant of search() that
//!           // stamps t1, t2, t3, t4 between stages — implemented via a
//!           // generic stage-callback or a dedicated `searchTimed`.
//!           timings.append(...)
//!           ranking = [r.doc_id for r in result]
//!           mrr_sum += tac.eval.metrics.mrrAt(ranking, qrels[q.qid], 10)
//!         report = tac.eval.latency.report(timings.items)
//!         mrr = mrr_sum / queries.len
//!         csv.writeRow(.{
//!           dataset: "msmarco-v1",
//!           kappa_c, kappa_d, alpha, n_queries: queries.len,
//!           quality: mrr,
//!           avg_total_ms: report.avg_total_ms,
//!           p50_total_ms: report.p50_total_ms,
//!           p95_total_ms: report.p95_total_ms,
//!           avg_gather_ms: report.avg_gather_ms,
//!           avg_prune_ms: report.avg_prune_ms,
//!           avg_table_ms: report.avg_table_ms,
//!           avg_refine_ms: report.avg_refine_ms,
//!           git_sha: build_options.git_sha,
//!           host: hostName(),
//!         })
//!   csv.close()
//!
//!   // Post-process: print the operating points closest to the paper targets.
//!   findOperatingPoint(csv.rows, target_mrr=39.0, summarise: true)
//!   findOperatingPoint(csv.rows, target_mrr=39.3, summarise: true)
//!
//! ---------------------------------------------------------------------------
//! Smoke variant for CI:
//!   - 100-doc synthetic fixture, n_queries=10, kappa_c=20, kappa_d=50, alpha=null.
//!   - Asserts: avg_total_ms is finite; mrr is in [0, 1]; ranking length == top_k
//!     (or fewer if fewer docs).
//!   - Runs in <1s — guards regressions on local pushes.

const std = @import("std");

pub fn main() !void {
    @panic("benchmarks/msmarco_v1.zig: not yet implemented (waits on #15, #19); see top-of-file pseudocode");
}
