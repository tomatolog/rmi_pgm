//
// Created by Giorgio Vinciguerra on 27/04/2020.
// Modified by Ryan Marcus, 29/04/2020.
//

#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include "rmis/wiki.h"
#include "rmis/fb.h"
#include "rmis/osm.h"
#include "rmis/books.h"
#include "PGM-index/include/pgm/pgm_index.hpp"
#include "RadixSpline/include/rs/multi_map.h"

#include "btree.h"
#include "set.h"

#define BRANCHLESS

#define BENCH_RMI
#define BENCH_PGM
#define BENCH_RS
#define BENCH_BTREE

#define DATA_WIKI
//#define DATA_FB
//#define DATA_OSM
//#define DATA_BOOKS

//uint64_t NUM_LOOKUPS = 10000000;
uint64_t NUM_LOOKUPS = 100;

using timer = std::chrono::high_resolution_clock;

std::vector<std::string> DATASET_NAMES = {
  "data/books_200M_uint64",
  "data/osm_cellids_200M_uint64",
  "data/wiki_ts_200M_uint64",
  "data/fb_200M_uint64"
};

// Function taken from https://github.com/gvinciguerra/rmi_pgm/blob/357acf668c22f927660d6ed11a15408f722ea348/main.cpp#L29.
// Authored by Giorgio Vinciguerra.
template<class ForwardIt, class T, class Compare = std::less<T>>
ForwardIt lower_bound_branchless(ForwardIt first, ForwardIt last, const T &value, Compare comp = Compare()) {
    auto n = std::distance(first, last);

    while (n > 1) {
        auto half = n / 2;
        //__builtin_prefetch(&*first + half / 2, 0, 0);
        //__builtin_prefetch(&*first + half + half / 2, 0, 0);
        first = comp(*std::next(first, half), value) ? first + half : first;
        n -= half;
    }

    return std::next(first, comp(*first, value));
}

template<typename T>
static std::vector<T> load_data(const std::string &filename) {
    std::vector<T> data;

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "unable to open " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read size.
    uint64_t size;
    in.read(reinterpret_cast<char *>(&size), sizeof(uint64_t));
    data.resize(size);

    // Read values.
    in.read(reinterpret_cast<char *>(data.data()), size * sizeof(T));
    in.close();

    return data;
}

// Use the below code to generate lookup keys drawn from the data keys. This ensures
// that every part of the dataset is accessed equally.
/*std::vector<std::pair<uint64_t, size_t>> generate_queries(std::vector<uint64_t>& dataset) {
  std::vector<std::pair<uint64_t, size_t>> results;
  results.reserve(NUM_LOOKUPS);

  std::mt19937 g(42);
  std::uniform_int_distribution<size_t> distribution(0, dataset.size());
  
  for (uint64_t i = 0; i < NUM_LOOKUPS; i++) {
    size_t idx = distribution(g);
    uint64_t key = dataset[idx];
    size_t correct_lb = std::distance(
      dataset.begin(),
      std::lower_bound(dataset.begin(), dataset.end(), key)
      );
    
    results.push_back(std::make_pair(key, correct_lb));
  }

  return results;
  }*/

// Use the below code to generate lookup keys drawn uniformly from the minimum
// and maximum data key. This can lead to misleading results when the underlying dataset
// has skew:
//   consider a dataset where most values range between 0 and 2^50, but the last
//   20 keys range between 2^51 and 2^64. Over 99% of uniformly drawn lookups will
//   only access the last 20 keys.
//
// When this is the case, all you are really testing is your CPU cache. The FB dataset
// demonstrates this.
std::vector<std::pair<uint64_t, size_t>> generate_queries(std::vector<uint64_t>& dataset) {
  std::vector<std::pair<uint64_t, size_t>> results;
  results.reserve(NUM_LOOKUPS);
  
  std::mt19937 g(42);
  std::uniform_int_distribution<uint64_t> distribution(dataset.front(), dataset.back() - 1);
  
  for (uint64_t i = 0; i < NUM_LOOKUPS; i++) {
    uint64_t key = distribution(g);
    size_t correct_lb = std::distance(dataset.begin(), std::lower_bound(dataset.begin(), dataset.end(), key));
    
    results.push_back(std::make_pair(key, correct_lb));
  }
  
  return results;
}

template<typename F, class V>
int64_t query_time(F f, const V &queries) {
  auto start = timer::now();
  
  uint64_t cnt = 0;
  for (auto &q : queries) {
    cnt += f(q.first, q.second);
  }
  
    auto stop = timer::now();

    std::cout << "query result " << cnt  << std::endl;

    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / queries.size();
}

inline void CheckValid ( size_t iPos, size_t iCorrect, const char * sName, uint64_t iKey, size_t iStart, size_t iEnd )
{
    if ( iPos!=iCorrect )
    {
      std::cerr << sName << " returned incorrect result for lookup key " << iKey << std::endl;
      std::cerr << "Start: " << iStart << " Stop: " << iEnd << " Correct: " << iCorrect << std::endl;
    }
}

void PrintSize ( size_t iSize )
{
    std::cout << std::fixed << std::setprecision(3) << ( (float)iSize / 1024.0f / 1024.0f ) << " Mb ("  << iSize << " bytes)";
}


static void ReportBuild ( const char * sName, int64_t tmBuild, size_t iSize )
{
    std::cout << sName << " construct \t\t" << std::fixed << std::setprecision(3) << (float)tmBuild / 1000000.0f << " sec" << std::endl;
    std::cout << sName << " index size \t\t";
    PrintSize ( iSize );
    std::cout << std::endl;
}

template<typename BTREE>
int64_t BenchBTree ( int iNodeSize, const std::vector<uint64_t> & dataset, const std::vector<std::pair<uint64_t, size_t>> & queries )
{
    int64_t tmBtree = 0;
    auto btree_build_start = timer::now();
    auto bidx = new BTREE();

    for ( auto & tItem : dataset )
        bidx->insert ( tItem );
    auto btree_build_stop = timer::now();
    auto btree_build_tm = std::chrono::duration_cast<std::chrono::microseconds>(btree_build_stop - btree_build_start).count();

    std::cout << "BTree leaf items \t" << iNodeSize << std::endl;
    ReportBuild ( "BTree", btree_build_tm, bidx->bytes_used() );

    tmBtree = query_time([bidx, &dataset](auto x, auto correct_idx)
    {
        //#ifdef BRANCHLESS
        //    auto lb_result = lower_bound_branchless(approx_range.first, approx_range.second, x);
        //#else
            auto lb_result = bidx->lower_bound(x);
        //#endif

        //size_t lb_position = lb_result - bidx->begin();
        return *lb_result;
    }, queries);

    delete ( bidx );

    return tmBtree;
}

template<int dataset_idx, uint64_t build_time, size_t rmi_size,
         uint64_t (* RMI_FUNC)(uint64_t, size_t*)>
void measure_perfomance() {
  std::string dataset_name = DATASET_NAMES[dataset_idx];
  std::cout << "Reading " << dataset_name << std::endl;
  auto dataset = load_data<uint64_t>(dataset_name);

  std::cout << "Generating queries..." << std::endl;
  std::vector<std::pair<uint64_t, size_t>> queries = generate_queries(dataset);
  
  std::cout << "Queries \t\t" << queries.size() << std::endl;
  std::cout << "Elements \t\t" << dataset.size() << std::endl;
  std::cout << "Size \t\t\t";
  PrintSize ( sizeof(uint64_t) * dataset.size() );
  std::cout << std::endl;
  ReportBuild ( "RMI", build_time, rmi_size );

  int64_t tmRmi = 0;
  // Test lookups for RMI.
#ifdef BENCH_RMI
  tmRmi = query_time([&dataset](auto x, auto correct_idx) {
    auto data_size_ = dataset.size();
    size_t error;
    uint64_t guess = RMI_FUNC(x, &error);
    uint64_t start = (guess < error ? 0 : guess - error);
    uint64_t stop = (guess + error >= data_size_ ? data_size_ : guess + error);

#ifdef BRANCHLESS
    auto lb_result = lower_bound_branchless(dataset.begin() + start, dataset.begin() + stop, x);
#else
    auto lb_result = std::lower_bound(dataset.begin() + start, dataset.begin() + stop, x);
#endif

    size_t lb_position = std::distance(dataset.begin(), lb_result);
    if (lb_position != correct_idx) {
      std::cerr << "RMI returned incorrect result for lookup key " << x << std::endl;
      std::cerr << "Start: " << start
                << " Stop: " << stop
                << " Correct: " << correct_idx << std::endl;
      std::cerr << "Start  key: " << dataset[start] << std::endl;
      std::cerr << "Stop   key: " << dataset[stop] << std::endl;
      std::cerr << "Stop+1 key: " << dataset[stop+1] << std::endl;
      exit(-1);
    }
    return lb_position;
  }, queries);
#endif

  int64_t tmPgm = 0;
  // Test lookups for PGM.
#ifdef BENCH_PGM
    auto pgm_build_start = timer::now();
    pgm::PGMIndex<uint64_t, 64> index(dataset);
    auto pgm_build_stop = timer::now();
    auto pgm_build_tm = std::chrono::duration_cast<std::chrono::microseconds>(pgm_build_stop - pgm_build_start).count();

    ReportBuild ( "PGM", pgm_build_tm, index.size_in_bytes() );

  tmPgm = query_time([&index, &dataset](auto x, auto correct_idx)
  {
    auto approx_range = index.search(x);

#ifdef BRANCHLESS
    auto lb_result = lower_bound_branchless(dataset.begin() + approx_range.lo, dataset.begin() + approx_range.hi, x);
#else
    auto lb_result = std::lower_bound(dataset.begin() + approx_range.lo, dataset.begin() + approx_range.hi, x);
#endif

    size_t lb_position = std::distance(dataset.begin(), lb_result);
    CheckValid ( lb_position, correct_idx, "PGM", x, approx_range.lo, approx_range.hi );
    return lb_position;
  }, queries);
#endif

    // Test lookups for RS
    auto minmax = std::minmax_element ( dataset.begin(), dataset.end() );
    auto rs_build_start = timer::now();
    rs::Builder<uint64_t> rsb(*minmax.first, *minmax.second);
    for ( const auto & tItem : dataset) rsb.AddKey ( tItem );
    rs::RadixSpline<uint64_t> ridx = rsb.Finalize();
    auto rs_build_stop = timer::now();
    auto rs_build_tm = std::chrono::duration_cast<std::chrono::microseconds>(rs_build_stop - rs_build_start).count();

    ReportBuild ( "RS", rs_build_tm, ridx.GetSize() );

    int64_t tmRs = 0;
  tmRs = query_time([&ridx, &dataset](auto x, auto correct_idx)
  {
    auto approx_range = ridx.GetSearchBound(x);

#ifdef BRANCHLESS
    auto lb_result = lower_bound_branchless(dataset.begin() + approx_range.begin, dataset.begin() + approx_range.end, x);
#else
    auto lb_result = std::lower_bound(dataset.begin() + approx_range.begin, dataset.begin() + approx_range.end, x);
#endif

    size_t lb_position = std::distance(dataset.begin(), lb_result);
    CheckValid ( lb_position, correct_idx, "RS", x, approx_range.begin, approx_range.end );
    return lb_position;
  }, queries);

  int64_t tmBtree1024 = 0;
  int64_t tmBtree512 = 0;
  int64_t tmBtree256 = 0;
  int64_t tmBtree128 = 0;
  int64_t tmBtree64 = 0;
    // Test lookups for BTree
#ifdef BENCH_BTREE
    tmBtree1024 = BenchBTree<btree::set<uint64_t, std::less<uint64_t>, std::allocator<uint64_t>, 1024>> ( 1024, dataset, queries );
    tmBtree512 = BenchBTree<btree::set<uint64_t, std::less<uint64_t>, std::allocator<uint64_t>, 512>> ( 512, dataset, queries );
    tmBtree256 = BenchBTree<btree::set<uint64_t, std::less<uint64_t>, std::allocator<uint64_t>, 256>> ( 256, dataset, queries );
    tmBtree128 = BenchBTree<btree::set<uint64_t, std::less<uint64_t>, std::allocator<uint64_t>, 128>> ( 128, dataset, queries );
    tmBtree64 = BenchBTree<btree::set<uint64_t, std::less<uint64_t>, std::allocator<uint64_t>, 64>> ( 64, dataset, queries );
#endif

  std::cout << dataset_name << ", ns per query" << std::endl
            << "RMI " << tmRmi << " ns,"
            << " PGM " << tmPgm << " ns,"
            << " RS " << tmRs << " ns,"
            << " Btree1024 " << tmBtree1024 << " ns" << " Btree512 " << tmBtree512 << " ns" << " Btree256 " << tmBtree256 << " ns" << " Btree128 " << tmBtree128 << " ns" << " Btree64 " << tmBtree64 << " ns"
            << std::endl;
}

int main(int argc, char **argv) {
    // load each RMI
    bool bLoaded = false;
#ifdef DATA_WIKI
    bLoaded = wiki::load ( "rmi_data/" );
    if ( bLoaded )
        measure_perfomance<2, wiki::RMI_SIZE, wiki::BUILD_TIME_NS, wiki::lookup>();
#endif
#ifdef DATA_FB
    bLoaded = fb::load ( "rmi_data/" );
    if ( bLoaded )
        measure_perfomance<3, fb::RMI_SIZE, fb::BUILD_TIME_NS, fb::lookup>();
#endif
#ifdef DATA_OSM
    bLoaded = osm::load ( "rmi_data/" );
    if ( bLoaded )
        measure_perfomance<1, osm::RMI_SIZE, osm::BUILD_TIME_NS, osm::lookup>();
#endif
#ifdef DATA_BOOKS
    bLoaded = books::load ( "rmi_data/" );
    if ( bLoaded )
        measure_perfomance<0, books::RMI_SIZE, books::BUILD_TIME_NS, books::lookup>();
#endif

    if ( !bLoaded )
    {
      std::cerr << "Failed to load RMIs" << std::endl;
      exit(-1);
    }
    
    return 0;
}
