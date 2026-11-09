#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <climits>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
using namespace std;

static int n       = 0;
static int k       = 0;
static int r       = 0;
static int total_n = 0;
static int dummy_n = 0;

static long long **W    = 0;
static int        *weight = 0;

static string Infile, Outfile_result, Outfile_detail, InstanceName;

static double eps     = 0.60;
static int    maxIter = 20;
static int    A       = 3;
static double alpha   = 0.10;
static double beta_   = 0.20;

static double gamma_  = 0.30;
static double rho     = 0.30;
static const double ETA_LOW  = 0.05;
static const double ETA_HIGH = 0.95;

static double timeLimit = 600.0;
static int    runs      = 1;
static bool   hasSeed   = false;
static long   seed      = 0;
static bool   verbose   = false;
static bool   selftest  = false;
static int    selftestTrials = 3000;
static bool   checkInvariants = false;

static double **Eta = 0;

struct Solution
{
    int       *g;
    int       *cnt;
    long long  value;
};

static Solution curS, localBestS, globalBestS, beginS;

static long long **delta = 0;

static inline int iabs(int x) { return x < 0 ? -x : x; }

static inline double rand01() { return rand() / ((double)RAND_MAX + 1.0); }

static void AllocSolution(Solution &s)
{
    s.g     = new int[n];
    s.cnt   = new int[k];
    s.value = 0;
}

static void CopySolution(Solution &dst, const Solution &src)
{
    memcpy(dst.g, src.g, sizeof(int) * n);
    memcpy(dst.cnt, src.cnt, sizeof(int) * k);
    dst.value = src.value;
}

static void Input()
{
    ifstream fin(Infile.c_str());
    if (fin.fail()) { cout << "error open" << endl; exit(1); }

    fin >> n;

    if (r > n)                             k = 1;
    else if (r < n && n < (3 * r + 3) / 2) k = 2;
    else if (r == 2 && n >= 9)             k = (int)ceil(double(2 * n) / 3) - 1;
    else if (r % 2 == 1)                   k = (int)floor(double(2 * n) / (r + 1));
    else
    {
        k = 1;
        for (int i = 1; i < 100; ++i)
        {
            int a = r / 2 + 1 + (r + 1) * i;
            if ((a - r) <= n && n <= a) { k = 2 * i + 1; break; }
        }
    }

    while (k * r < n) ++k;

    total_n = k * r;
    dummy_n = total_n - n;

    W = new long long*[n];
    for (int i = 0; i < n; ++i) W[i] = new long long[n];
    weight = new int[n];

    cout << "vaildfacality:" << n << "  totalfacality:" << total_n << endl;
    cout << "rows:" << r << "  column:" << k << endl;

    vector<long long> rest;
    {
        long long tok;
        while (fin >> tok) rest.push_back(tok);
    }
    fin.close();

    const size_t need = (size_t)n + (size_t)n * (size_t)n;
    if (rest.size() < need)
        {
            fprintf(stderr,
                    "error: %s: the file holds %zu numeric tokens after n, but the format "
                    "needs %zu (n weights + n x n matrix, n = %d).\n"
                    "       Short by %zu - refusing to solve a misaligned matrix.\n",
                    Infile.c_str(), rest.size(), need, n, need - rest.size());
            exit(1);
        }

    for (int i = 0; i < n; ++i) weight[i] = (int)rest[i];

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const long long w = rest[(size_t)n + (size_t)i * (size_t)n + (size_t)j];
            if (i > j) W[i][j] = W[j][i];
            else       W[i][j] = w;
        }

    bool nonUnit = false;
    for (int i = 0; i < n; ++i) if (weight[i] != 1) { nonUnit = true; break; }
    if (nonUnit)
        fprintf(stderr,
                "warning: %s carries a per-facility weight vector that the MREFLP "
                "model of Eq. (1) does not use; it is ignored (the original also "
                "read weight[] and never referenced it).\n", Infile.c_str());
}

static void AssignMemory()
{
    AllocSolution(curS);
    AllocSolution(localBestS);
    AllocSolution(globalBestS);
    AllocSolution(beginS);

    delta = new long long*[n];
    for (int i = 0; i < n; ++i) delta[i] = new long long[k];

    Eta = new double*[n];
    for (int i = 0; i < n; ++i) Eta[i] = new double[k]();
}

static void DestroyMemory()
{
    for (int i = 0; i < n; ++i) delete[] W[i];
    delete[] W;
    delete[] weight;
    delete[] curS.g;        delete[] curS.cnt;
    delete[] localBestS.g;  delete[] localBestS.cnt;
    delete[] globalBestS.g; delete[] globalBestS.cnt;
    delete[] beginS.g;      delete[] beginS.cnt;
    for (int i = 0; i < n; ++i) delete[] delta[i];
    delete[] delta;
    for (int i = 0; i < n; ++i) delete[] Eta[i];
    delete[] Eta;
}

static long long ObjectiveFromGroups(const int *g)
{
    long long f = 0;
    for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v)
            f += W[u][v] * iabs(g[u] - g[v]);
    return f;
}

static void InitDelta(const Solution &S)
{
    for (int v = 0; v < n; ++v)
    {
        const int gv = S.g[v];
        for (int j = 0; j < k; ++j)
        {
            if (j == gv) { delta[v][j] = 0; continue; }
            long long d = 0;
            for (int u = 0; u < n; ++u)
            {
                if (u == v) continue;
                d += (long long)(iabs(j - S.g[u]) - iabs(gv - S.g[u])) * W[v][u];
            }
            delta[v][j] = d;
        }
    }
}

static void OneMoveDeltaUpdate(const Solution &S, int v, int from, int to)
{
    for (int u = 0; u < n; ++u)
    {
        if (u == v) continue;
        const long long wuv = W[u][v];
        if (wuv == 0) continue;
        const int gu = S.g[u];
        const long long base = (long long)(iabs(gu - from) - iabs(gu - to));
        for (int q = 0; q < k; ++q)
            delta[u][q] += (long long)(iabs(q - to) - iabs(q - from) + base) * wuv;
    }
    const int gv = S.g[v];
    for (int j = 0; j < k; ++j)
    {
        if (j == gv) { delta[v][j] = 0; continue; }
        long long d = 0;
        for (int u = 0; u < n; ++u)
        {
            if (u == v) continue;
            d += (long long)(iabs(j - S.g[u]) - iabs(gv - S.g[u])) * W[v][u];
        }
        delta[v][j] = d;
    }
}

static long long ll_abs(long long x) { return x < 0 ? -x : x; }

static bool DeltaMatchesScratch(const Solution &S, long long &maxErr)
{
    long long *row = new long long[k];
    bool ok = true;
    maxErr = 0;
    for (int v = 0; v < n; ++v)
    {
        const int gv = S.g[v];
        for (int j = 0; j < k; ++j)
        {
            long long d = 0;
            if (j != gv)
                for (int u = 0; u < n; ++u)
                {
                    if (u == v) continue;
                    d += (long long)(iabs(j - S.g[u]) - iabs(gv - S.g[u])) * W[v][u];
                }
            row[j] = d;
        }
        for (int j = 0; j < k; ++j)
            if (delta[v][j] != row[j])
            { ok = false; maxErr = max(maxErr, ll_abs(delta[v][j] - row[j])); }
    }
    delete[] row;
    return ok;
}

static void CheckInvariants(const Solution &S, const char *where)
{
    const long long truth = ObjectiveFromGroups(S.g);
    long long dErr = 0;
    const bool dOk = DeltaMatchesScratch(S, dErr);

    int *c = new int[k];
    for (int j = 0; j < k; ++j) c[j] = 0;
    for (int f = 0; f < n; ++f) c[S.g[f]]++;
    bool cOk = true;
    for (int j = 0; j < k; ++j) if (c[j] != S.cnt[j]) cOk = false;
    delete[] c;

    if (truth != S.value || !dOk || !cOk)
        printf("INVARIANT VIOLATION (%s): value=%lld truth=%lld deltaOk=%d (err %lld) cntOk=%d\n",
               where, S.value, truth, (int)dOk, dErr, (int)cOk);
}

static void InitEta()
{
    for (int v = 0; v < n; ++v)
        for (int j = 0; j < k; ++j)
            Eta[v][j] = 1.0 / k;
}

static void LearningDrivenConstruction(Solution &S)
{
    for (int j = 0; j < k; ++j) S.cnt[j] = 0;

    int *pi = new int[n];
    for (int i = 0; i < n; ++i) pi[i] = i;
    int npi = n;

    int *CG  = new int[k];
    int nCG  = 0;
    for (int j = 0; j < k; ++j) CG[nCG++] = j;

    while (npi > 0)
    {

        const int idx = rand() % npi;
        const int v   = pi[idx];
        pi[idx] = pi[--npi];

        int group;
        if (rand01() < eps)
        {
            group = CG[0];
            for (int t = 1; t < nCG; ++t)
                if (Eta[v][CG[t]] > Eta[v][group]) group = CG[t];
        }
        else
        {
            group = CG[rand() % nCG];
        }

        S.g[v] = group;
        S.cnt[group]++;
        if (S.cnt[group] == r)
        {
            for (int t = 0; t < nCG; ++t)
                if (CG[t] == group) { CG[t] = CG[--nCG]; break; }
        }
    }

    S.value = ObjectiveFromGroups(S.g);
    delete[] pi;
    delete[] CG;
}

static void OneMoveTabuSearch(Solution &S, int depth)
{
    CopySolution(localBestS, S);
    InitDelta(S);

    int *tabu = new int[n];
    for (int i = 0; i < n; ++i) tabu[i] = 0;

    int NoImprove = 0;
    int iter      = 0;

    while (NoImprove < depth)
    {
        ++iter;

        bool      found    = false;
        long long bestGain = 0;
        int       bestV = -1, bestJ = -1;

        for (int v = 0; v < n; ++v)
        {
            const int gv = S.g[v];
            for (int j = 0; j < k; ++j)
            {
                if (j == gv)      continue;
                if (S.cnt[j] >= r) continue;

                const long long gain = delta[v][j];
                const bool notTabu   = (iter >= tabu[v]);
                const bool aspiration = (S.value + gain < localBestS.value);
                if (!(notTabu || aspiration)) continue;

                if (!found || gain < bestGain)
                { found = true; bestGain = gain; bestV = v; bestJ = j; }
            }
        }

        if (!found) { ++NoImprove; continue; }

        const int from = S.g[bestV];
        S.cnt[from]--;
        S.g[bestV]  = bestJ;
        S.cnt[bestJ]++;
        S.value    += bestGain;
        tabu[bestV] = iter + (rand() % A + 1);

        OneMoveDeltaUpdate(S, bestV, from, bestJ);

        if (S.value < localBestS.value) { NoImprove = 0; CopySolution(localBestS, S); }
        else                            ++NoImprove;
    }

    delete[] tabu;

    CopySolution(S, localBestS);
    InitDelta(S);
}

static void SwapDescentSearch(Solution &S)
{
    bool improved = true;
    while (improved)
    {
        improved = false;

        bool      found    = false;
        long long bestGain = 0;
        int       bu = -1, bw = -1;

        for (int u = 0; u < n; ++u)
        {
            for (int v = u + 1; v < n; ++v)
            {
                const int gu = S.g[u], gv = S.g[v];
                if (gu == gv) continue;
                const long long gain = delta[u][gv] + delta[v][gu]
                                     + 2LL * W[u][v] * iabs(gu - gv);
                if (!found || gain < bestGain)
                { found = true; bestGain = gain; bu = u; bw = v; }
            }
        }

        if (found && bestGain < 0)
        {
            const int gu = S.g[bu], gv = S.g[bw];
            S.value += bestGain;

            S.g[bu] = gv; S.cnt[gu]--; S.cnt[gv]++;
            OneMoveDeltaUpdate(S, bu, gu, gv);

            S.g[bw] = gu; S.cnt[gv]--; S.cnt[gu]++;
            OneMoveDeltaUpdate(S, bw, gv, gu);

            improved = true;
        }
    }
}

static void TwoPhaseLocalSearch(Solution &S)
{
    OneMoveTabuSearch(S, maxIter);
    if (checkInvariants) CheckInvariants(S, "after tabu search");

    SwapDescentSearch(S);
    if (checkInvariants) CheckInvariants(S, "after swap descent");
}

static void UpdateEta(const Solution &S0, const Solution &SB)
{
    for (int v = 0; v < n; ++v)
    {
        const int i = S0.g[v];
        const int j = SB.g[v];

        if (i == j)
        {

            for (int p = 0; p < k; ++p)
                Eta[v][p] = (p == i) ? (alpha + (1.0 - alpha) * Eta[v][p])
                                     : ((1.0 - alpha) * Eta[v][p]);
        }
        else
        {

            for (int p = 0; p < k; ++p)
            {
                if (p == i)
                    Eta[v][p] = (1.0 - beta_) * (1.0 - gamma_) * Eta[v][p];
                else if (p == j)
                    Eta[v][p] = gamma_ + (1.0 - gamma_) * beta_ / (k - 1)
                              + (1.0 - beta_) * (1.0 - gamma_) * Eta[v][p];
                else
                    Eta[v][p] = (1.0 - gamma_) * beta_ / (k - 1)
                              + (1.0 - beta_) * (1.0 - gamma_) * Eta[v][p];
            }
        }
    }
}

static void SmoothEta()
{
    for (int v = 0; v < n; ++v)
        for (int p = 0; p < k; ++p)
        {
            if      (Eta[v][p] > ETA_HIGH) Eta[v][p] = (1.0 - rho) * Eta[v][p];
            else if (Eta[v][p] < ETA_LOW)  Eta[v][p] = rho + (1.0 - rho) * Eta[v][p];
        }
}

static double find_best = 0.0;

static long long GeneralAlgo()
{
    InitEta();
    globalBestS.value = LLONG_MAX;
    bool haveBest = false;
    find_best = 0.0;

    const clock_t start = clock();
    long long iterations = 0;

    do
    {
        LearningDrivenConstruction(curS);
        CopySolution(beginS, curS);

        TwoPhaseLocalSearch(curS);

        if (!haveBest || curS.value < globalBestS.value)
        {
            haveBest = true;
            CopySolution(globalBestS, curS);
            find_best = (double)(clock() - start) / CLOCKS_PER_SEC;
        }

        UpdateEta(beginS, curS);
        SmoothEta();
        ++iterations;

        if (verbose)
            cout << "after local:" << curS.value
                 << " global:" << globalBestS.value << endl;
    }
    while ((double)(clock() - start) / CLOCKS_PER_SEC < timeLimit);

    if (verbose) cout << "iterations:" << iterations << endl;
    return globalBestS.value;
}

static void BuildLayout(const Solution &S, int *loc)
{
    int nextDummy = n;
    for (int j = 0; j < k; ++j)
    {
        int t = 0;
        for (int f = 0; f < n; ++f)
            if (S.g[f] == j) loc[j * r + t++] = f;
        for (; t < r; ++t) loc[j * r + t] = nextDummy++;
    }
}

static void Output()
{
    if (!Outfile_result.empty())
    {
        FILE *fp = fopen(Outfile_result.c_str(), "a+");
        if (fp)
        {
            fprintf(fp, "%s\t%lld\t%lf\t\n", InstanceName.c_str(),
                    globalBestS.value, find_best);
            fclose(fp);
        }
        else fprintf(stderr, "warning: cannot write %s\n", Outfile_result.c_str());
    }
    if (!Outfile_detail.empty())
    {
        int *loc = new int[total_n];
        BuildLayout(globalBestS, loc);
        FILE *fp = fopen(Outfile_detail.c_str(), "a+");
        if (fp)
        {
            fprintf(fp, "instance:%s n:%d total_n:%d\n", InstanceName.c_str(), n, total_n);
            for (int i = 0; i < total_n; ++i) fprintf(fp, " %d", loc[i]);
            fprintf(fp, "\n");
            fclose(fp);
        }
        else fprintf(stderr, "warning: cannot write %s\n", Outfile_detail.c_str());
        delete[] loc;
    }
}

static bool ParseNumber(const char *s, double &out)
{
    if (!s || !*s) return false;
    char *end = 0;
    out = strtod(s, &end);
    return end != s && *end == '\0';
}

static int SelfTest(int trials)
{
    printf("selftest: n=%d k=%d r=%d trials=%d\n", n, k, r, trials);

    Solution T;
    AllocSolution(T);
    LearningDrivenConstruction(T);
    InitDelta(T);

    int nMove = 0, nSwap = 0;
    int badMove = 0, badSwap = 0, badDelta = 0, badValue = 0;
    long long worstMove = 0, worstSwap = 0, worstDelta = 0, worstValue = 0;

    T.value = ObjectiveFromGroups(T.g);

    for (int t = 0; t < trials; ++t)
    {

        if (k > 1)
        {
            const int v = rand() % n;
            const int j = rand() % k;
            if (j != T.g[v])
            {
                const long long predicted = delta[v][j];
                const long long before    = ObjectiveFromGroups(T.g);
                const int from = T.g[v];
                T.cnt[from]--; T.g[v] = j; T.cnt[j]++;
                const long long gained = ObjectiveFromGroups(T.g) - before;
                if (predicted != gained)
                { ++badMove; worstMove = max(worstMove, ll_abs(predicted - gained)); }
                T.value += predicted;
                if (T.value != ObjectiveFromGroups(T.g))
                { ++badValue; worstValue = max(worstValue, ll_abs(T.value - ObjectiveFromGroups(T.g))); }
                OneMoveDeltaUpdate(T, v, from, j);
                long long e;
                if (!DeltaMatchesScratch(T, e)) { ++badDelta; worstDelta = max(worstDelta, e); }
                ++nMove;
            }
        }

        if (k > 1)
        {
            const int bu = rand() % n, bw = rand() % n;
            if (bu != bw && T.g[bu] != T.g[bw])
            {
                const int gu = T.g[bu], gv = T.g[bw];
                const long long predicted =
                    delta[bu][gv] + delta[bw][gu] + 2LL * W[bu][bw] * iabs(gu - gv);
                const long long before = ObjectiveFromGroups(T.g);

                T.g[bu] = gv; T.cnt[gu]--; T.cnt[gv]++; OneMoveDeltaUpdate(T, bu, gu, gv);
                T.g[bw] = gu; T.cnt[gv]--; T.cnt[gu]++; OneMoveDeltaUpdate(T, bw, gv, gu);

                const long long gained = ObjectiveFromGroups(T.g) - before;
                if (predicted != gained)
                { ++badSwap; worstSwap = max(worstSwap, ll_abs(predicted - gained)); }
                T.value += predicted;
                if (T.value != ObjectiveFromGroups(T.g))
                { ++badValue; worstValue = max(worstValue, ll_abs(T.value - ObjectiveFromGroups(T.g))); }
                long long e;
                if (!DeltaMatchesScratch(T, e)) { ++badDelta; worstDelta = max(worstDelta, e); }
                ++nSwap;
            }
        }
    }

    delete[] T.g; delete[] T.cnt;

    printf("  OneMove checks : %d, mismatches %d (worst |err| %lld)\n",
           nMove, badMove, worstMove);
    printf("  swap   checks : %d, mismatches %d (worst |err| %lld)\n",
           nSwap, badSwap, worstSwap);
    printf("  delta  rebuilds: %d failed (worst |err| %lld)\n", badDelta, worstDelta);
    printf("  value  updates : %d, mismatches %d (worst |err| %lld)\n",
           nMove + nSwap, badValue, worstValue);

    const bool pass = (badMove == 0 && badSwap == 0 && badDelta == 0 && badValue == 0);
    printf("selftest: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s <instance> <out_result> <out_detail> <r> [options]\n"
        "  r : max facilities per group (the paper's r, the original's \"rows\")\n"
        "options:\n"
        "  --eps E          greediness ratio                (Table 1: 0.6)\n"
        "  --maxIter N      tabu search depth               (Table 1: 20)\n"
        "  --A N            tabu tenure threshold           (Table 1: 3)\n"
        "  --alpha V        reward factor                   (Table 1: 0.1)\n"
        "  --beta V         penalization factor             (Table 1: 0.2)\n"
        "  --gamma V        compensation factor             (Table 1: 0.3)\n"
        "  --rho V          smoothing coefficient           (Table 1: 0.3)\n"
        "  --seed S         RNG seed (default: clock, once, at start)\n"
        "  --timeLimit SEC  cutoff in seconds               (paper: 600)\n"
        "  --runs K         independent runs, K seeds        (paper: 1 or 10)\n"
        "  --verbose        print the per-iteration trace\n"
        "  --check          verify value/delta invariants after every phase\n"
        "  --selftest [N]   validate Eqs. (8),(9),(10),(13) numerically, then exit\n",
        prog);
}

int main(int argc, char **argv)
{
    if (argc == 1)
    {

        Infile          = "/Users/qiyue/Desktop/test.txt";
        Outfile_result  = "/Users/qiyue/Desktop/resultlearn.txt";
        Outfile_detail  = "/Users/qiyue/Desktop/detaillearn.txt";
        r = 3;
    }
    else if (argc < 5)
    {
        usage(argv[0]);
        return 1;
    }
    else
    {
        Infile         = argv[1];
        Outfile_result = argv[2];
        Outfile_detail = argv[3];
        r              = atoi(argv[4]);
        if (r <= 0) { fprintf(stderr, "invalid r: %s\n", argv[4]); return 1; }
    }

    for (int i = 5; i < argc; ++i)
    {
        const string o = argv[i];

        const bool wantsValue = (o == "--eps"     || o == "--maxIter" || o == "--A"    ||
                                 o == "--alpha"   || o == "--beta"    || o == "--gamma" ||
                                 o == "--rho"     || o == "--seed"    ||
                                 o == "--timeLimit" || o == "--runs");
        if (wantsValue)
        {
            double v;
            if (i + 1 >= argc || !ParseNumber(argv[i + 1], v))
            {
                fprintf(stderr, "error: option %s requires a numeric value%s%s\n",
                        o.c_str(), (i + 1 < argc) ? ", got '" : "",
                        (i + 1 < argc) ? argv[i + 1] : "");
                if (i + 1 < argc) fprintf(stderr, "'\n");
                return 1;
            }
            ++i;
            if      (o == "--eps")       eps       = v;
            else if (o == "--maxIter")   maxIter   = (int)v;
            else if (o == "--A")         A         = (int)v;
            else if (o == "--alpha")     alpha     = v;
            else if (o == "--beta")      beta_     = v;
            else if (o == "--gamma")     gamma_    = v;
            else if (o == "--rho")       rho       = v;
            else if (o == "--seed")    { seed      = (long)v; hasSeed = true; }
            else if (o == "--timeLimit") timeLimit = v;
            else if (o == "--runs")      runs      = (int)v;
            continue;
        }

        if      (o == "--verbose") verbose         = true;
        else if (o == "--check")   checkInvariants = true;
        else if (o == "--selftest")
        {
            selftest = true;
            double v;
            if (i + 1 < argc && ParseNumber(argv[i + 1], v)) { selftestTrials = (int)v; ++i; }
        }
        else fprintf(stderr, "warning: ignoring unrecognised argument '%s'\n", o.c_str());
    }
    if (maxIter < 1) maxIter = 1;
    if (A < 1) A = 1;
    if (runs < 1) runs = 1;
    if (selftestTrials < 1) selftestTrials = 1;

    if (argc > 1)
    {
        InstanceName = Infile.substr(Infile.find_last_of("/\\") + 1);
        const size_t dot = InstanceName.find_last_of(".");
        if (dot != string::npos) InstanceName = InstanceName.substr(0, dot);
    }
    else
    {
        InstanceName = "test";
    }

    if (!hasSeed) { seed = (long)time(NULL); hasSeed = true; }
    srand((unsigned)seed);

    Input();
    AssignMemory();

    if (selftest)
    {

        InitEta();
        const int rc = SelfTest(selftestTrials);
        DestroyMemory();
        return rc;
    }

    for (int run = 0; run < runs; ++run)
    {
        srand((unsigned)(seed + run));
        const long long best = GeneralAlgo();

        int *loc = new int[total_n];
        BuildLayout(globalBestS, loc);
        cout << "value:" << best << "time:" << find_best << endl;
        for (int i = 0; i < total_n; ++i) cout << " " << loc[i];
        cout << endl;
        delete[] loc;

        Output();
    }

    DestroyMemory();
    return 0;
}
