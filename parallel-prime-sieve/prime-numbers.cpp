// prime-numbers.cpp, program (UTF-8)
//
// Compute prime numbers using an Eratosthenes sieve.
// This is a parallel implementation using OpenMP (>=3.0)
//
// Copyright (C) 2013-2015 Thomas Zoschke
// email zoschke dot thomas at google mail (Don't like beeing scanned by bots;)
// ---------------------------------------------------------------------------
/* 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin St - Fifth Floor, Boston, MA 02110-1301 USA.
 * 
 */
// ---------------------------------------------------------------------------
//
// The sieve is initialy zeroed, meaning all numbers could possibly be prime.
// Sieving finds composit numbers, ruling them out as prime numbers,
// those will be coded as ONE bits in the sieve, but before saving the
// sieve is negated, so the ONEs stand for the primes in the persistant form.
// This is a sieve of odds only (see below).
//
// ---------------------------------------------------------------------------

#include <omp.h>
#include <cstdio>
#include <climits>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <initializer_list>
#include <iostream>
#include <iomanip>
#include <utility>

#ifdef __unix
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#else
#error TODO: Porting.
#endif

using namespace std;

#ifdef DEBUG
#define ASSERT( x )  assert(x)
#define VERBOSE 1
#else
#define ASSERT( x )
#define VERBOSE 0
#endif

#define WORD_BITS       64              /* we're using uint64_t "words" */
#define ULL		unsigned long long

#ifndef L2_CACHE
#define L2_CACHE        2*1024*1024     /* size of CPU's L2 cache in bytes */
#endif


// IS_COMPOSITE tests for a one bit in the sieve,
// meaning a composit number found while sieving
#define IS_COMPOSITE(SIEVE,N) \
    ({ \
        unsigned __word_num, __bit__pos; \
        __word_num = ((N)>>1)/WORD_BITS, __bit__pos = ((N)>>1)%WORD_BITS; \
        SIEVE[__word_num] & (1uLL<<__bit__pos); \
    })

// a ZERO bit in a sufficiently sieved out sieve range denotes a prime number
// detected by the IS_PRIME macro.
#define IS_PRIME(SIEVE,N) \
    (!IS_COMPOSITE(SIEVE,N))

// set a sieve bit in SIEVE to denote a composit number N
#define SET_COMPOSITE(SIEVE,N) \
    ({ \
        unsigned __word_num, __bit__pos; \
        __word_num = ((N)>>1)/WORD_BITS, __bit__pos = ((N)>>1)%WORD_BITS; \
        SIEVE[__word_num] |= (1uLL<<__bit__pos);   \
    })

// unset (i.e. mask) a bit in SIEVE to denote a prime number N
#define SET_PRIME(SIEVE,N) \
    ({ \
        unsigned __word_num, __bit__pos; \
        __word_num = ((N)>>1)/WORD_BITS, __bit__pos = ((N)>>1)%WORD_BITS; \
        SIEVE[__word_num] &=~(1uLL<<__bit__pos);   \
    })
    
/* BITPOS  -->  NUMBER
 * 0            1
 * 1            3
 * 2            5
 * 3            7
 * ...
 */
#define BITPOS_TO_NUMBER(B) \
    (1uLL+(uint64_t(B)<<1uLL))



__extension__  __attribute__ ((noreturn))
void die (const char* last_msg)
{
    std::clog << last_msg << std::endl;
    exit (1);
}


uint32_t next_prime (uint64_t* sieve, uint32_t number)
{
    ASSERT(number&1);
    uint32_t half = number>>1;
again:
    ++ half;
    {   unsigned word_num, bit__pos;
        word_num = half/WORD_BITS, bit__pos = half%WORD_BITS;
        if (sieve[word_num] & (1uLL<<bit__pos)) goto again;
    }
    return 1+half+half;
}


inline
uint64_t largest_number_in_block (uint64_t block_size_in_words)
{
    return BITPOS_TO_NUMBER((block_size_in_words-1) * WORD_BITS) + BITPOS_TO_NUMBER(WORD_BITS-1) - 1;
}


void init_raster (uint64_t* raster, std::initializer_list<uint32_t> lst, int32_t raster_size)
// make a raster filled with ALL odd multiples in lst
{
    uint32_t limit = largest_number_in_block (raster_size);
    for (uint32_t i : lst) {
        // ASSERT( is_prime(i) );
        //   THIS IS PURELY CONCEPTIONELL, THE ASSERT CANNOT BE DONE RIGHT NOW,
        //   AS THE PRIMES ARE WORK IN PROGRESS!
        uint32_t distance = i+i;
        uint32_t multiple = i; // start with FIRST occurance of number,
                               // this MUST NOT skip anything as raster repeats in sieve!!!
        do {
            SET_COMPOSITE(raster,multiple);
            multiple += distance;
        } while (multiple<=limit);
    }
}


uint64_t* create_huge_memory_block_sieved_with_small_primes
    (
        uint64_t        wanted_size_in_words,
        uint64_t&       largest_number_in_sieve         // out
    )
{
    typedef char* chp;
    enum { 
        raster_sz = 3*5*7*11*13*17 // 255255 64-Bit words, 2042040 bytes, ca. 97.372% L2_CACHE
    }; 
    uint64_t* result = NULL;
    uint64_t q, r;
    q = wanted_size_in_words / raster_sz, r = wanted_size_in_words % raster_sz;
    // Just enlarge this hard limit if you want several TB of primes
    if (q*raster_sz*8 > 1000uLL*1000*1000*1000) die("too big. (1)");
    largest_number_in_sieve = largest_number_in_block (wanted_size_in_words);
    // NOTE: If you enlarge this hard limit you need to change the prime number
    //  caching vector template from uint32_t to uint64_t.
    if (sqrt(largest_number_in_sieve)>=0xFFFFffffuLL) die("too big. (2)");
    
    result =  (uint64_t*)   mmap (nullptr, sizeof(uint64_t)*max((ULL)wanted_size_in_words,(ULL)raster_sz),
                                  PROT_READ|PROT_WRITE,
                                  MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, /*fd*/-1, /*offset*/0);
    
    if (MAP_FAILED==result) die("mmap failed.");
    init_raster (result, {3,5,7,11,13,17}, raster_sz);
    for (uint32_t i=1; i<q; ++i)
        memcpy(chp(result)+sizeof(uint64_t)*i*raster_sz,chp(result),sizeof(uint64_t)*raster_sz);
    if (r && q)
        memcpy(chp(result)+sizeof(uint64_t)*q*raster_sz,chp(result),sizeof(uint64_t)*r);
    
                // ^- dest                                  ^- src      ^- size
    
    SET_COMPOSITE(result,1);
    SET_PRIME(result,3);        
    SET_PRIME(result,5); 
    SET_PRIME(result,7); 
    SET_PRIME(result,11);
    SET_PRIME(result,13);
    SET_PRIME(result,17);
    // now it is safe to test IS_COMPOSITE(result,X) for all odd X in [1..17²] = [1..289]
    return result;
}


uint32_t hamming_weight (uint64_t x) // also known as popcount or population count
{
    x = (x & 0x5555555555555555uLL) + ((x >> 1) & 0x5555555555555555uLL);
    x = (x & 0x3333333333333333uLL) + ((x >> 2) & 0x3333333333333333uLL);
    x = (x & 0x0F0F0F0F0F0F0F0FuLL) + ((x >> 4) & 0x0F0F0F0F0F0F0F0FuLL);
    return (x * 0x0101010101010101uLL) >> 56;
}


uint64_t count_primes_and_negate (uint64_t *const sieve, const uint64_t from, const uint64_t to)
// bounds [from..to] are inclusive
// bounds have to be odd and byte aligned
// caller has to do the bounds check calculation for sieve[]
// sieve bits get negated during counting so that the primes become the one bits
{
    ASSERT( to!=0 );
    ASSERT( from!=~0uLL );
    ASSERT( from<=to );
    uint64_t count = 0;
    if (from&to&1 && 0==(from>>1)%WORD_BITS && (WORD_BITS-1)==(to>>1)%WORD_BITS) { // aligned and odd, this is fast
        std::ptrdiff_t first_word = (from>>1)/WORD_BITS;
        std::ptrdiff_t last_word = (to>>1)/WORD_BITS; // this is inclusive
        for (uint64_t* p = sieve+first_word; p != sieve+(last_word+1); ++p)
            count += hamming_weight ( *p=~*p );
    }
    else {
        die("unaligned or even not yet handled");
    }
    if (from<=2 && 2<=to) { // two in range, cheat in the only even prime into the count
        ++count;
    }
    return count;
}


uint64_t eratosthenes_odd_single_block_seq
    (
    uint64_t* const        sieve,             // sieve array, in-out
    const uint64_t         from,              // inclusive range [from..to] lower bound
    const uint64_t         to,                // inclusive range [from..to] upper bound
    std::vector<uint32_t>& sieve_prime,       // sieve prime cache vector, in-out, non empty
    bool&                  parallel_OK,       // flag, in-out
    uint32_t               isqrt_max_num      // square root of topmost block upper inclusive bound rounded down
    )
// sieve a block
// returns count of prime numbers in range [from..to] (inclusive range)
{
    uint32_t limit = (int32_t) sqrt (to);
    auto iter = sieve_prime.begin ();
    auto iter_end = sieve_prime.end ();
    uint32_t prime_number;
    for (;;) {
        prime_number = iter!=iter_end ? *iter++ : next_prime (sieve, prime_number);
        if (prime_number>limit) break;
        uint64_t multiple = uint64_t(prime_number)*uint64_t(prime_number);
        int32_t distance = int32_t(prime_number)+int32_t(prime_number);
        int64_t delta = int64_t(from)-int64_t(multiple);
        int64_t N;
        if (0<delta) { // multiple < from, not yet inside block
            N = delta/distance;
            if (delta%distance) ++N;
            multiple += N*distance;
        }
        // run the sieve:
        while (multiple<=to) {
            SET_COMPOSITE(sieve,multiple);
            multiple += distance;
        }
    }
    // fill prime number caching vector:
    if (!parallel_OK) {
        // find first prime that goes into vector
        prime_number = next_prime (sieve, *(sieve_prime.rbegin ()));
        while ( prime_number <= to ) {
            sieve_prime.push_back (prime_number);
            if (prime_number > isqrt_max_num){
                parallel_OK = true;
                break;
            }
            prime_number = next_prime (sieve, prime_number);
        }
    }
    return count_primes_and_negate (sieve, from, to);
}


uint64_t eratosthenes_odd_single_block_par
    (
    uint64_t* const              sieve,        // sieve array, in-out   (in only when counting)
    const uint64_t               from,         // inclusive range [from..to] lower bound
    const uint64_t               to,           // inclusive range [from..to] upper bound
    const std::vector<uint32_t>& sieve_prime   // sieve prime cache vector, in (large enough)
    )
// sieve a block
// returns count of prime numbers in range [from..to] (inclusive range)
{
    uint32_t limit = (int32_t) sqrt (to);
    auto iter = sieve_prime.begin ();
    for (;;) {
        uint32_t prime_number = *iter++;
        if (prime_number>limit) break;
        uint64_t multiple = uint64_t(prime_number)*uint64_t(prime_number);
        int32_t distance = int32_t(prime_number)+int32_t(prime_number);
        int64_t delta = int64_t(from)-int64_t(multiple);
        int64_t N;
        if (0<delta) { // multiple < from, not yet inside block
            N = delta/distance;
            if (delta%distance) ++N;
            multiple += N*distance;
        }
        // run the sieve:
        while (multiple<=to) {
            SET_COMPOSITE(sieve,multiple);
            multiple += distance;
        }
    }
    return count_primes_and_negate (sieve, from, to);
}


/*
    \mathrm{Ei}(x) = \gamma+\ln |x| + \sum_{k=1}^{\infty} \frac{x^k}{k\; k!} \qquad x \neq 0 
    
    x^1/1 + x^2/(2*1*2) + x^3/(3*1*2*3) ...
*/
double exponential_integral (double x)
{
    double result = 0.577215664901532 + log (fabs (x));
    double factor = x;
    double sum = 0.0;
    uint32_t k = 1;
    do {
        sum += factor/k;
        ++k;
        factor *= (x/k);
        //printf ( "%u, sum=%G, factor=%G\n", k, sum, factor );
    } while ( factor/k > 1e-6 );
    return result+sum;
}


double logarithmic_integral (double x)
// logarithmic_integral (x) = integrate (1/ln (t), t=0..x)
{
    return exponential_integral ( log (x));
}


uint32_t guess_prim_num_count (uint64_t from, uint64_t to)
{
    if ( from < to && 1 < from ) {
        return logarithmic_integral (to) - logarithmic_integral (from);
    }
    return 0;
}


#define SQUARE_64( x ) ((unsigned long long)(x)*(unsigned long long)(x))


uint64_t eratosthenes_odd_blockwise 
    (
        uint64_t* sieve,
        uint64_t min_num,
        uint64_t max_num,
        uint32_t lps_prime,
        uint64_t n_slice
    )
// complete sieve process
// returns: prime number count in [min_num..max_num]
{
    ASSERT( min_num <= max_num);
    ASSERT( sieve );      // valid pointer to array large enough
    ASSERT( min_num & 1); // we are sieving ODD ONLY!!!
    ASSERT( max_num & 1); // we are sieving ODD ONLY!!!
    ASSERT( n_slice );    // 0 would not advance
    ASSERT( 0 == (n_slice&1) ); // advance needs to be even so that next start is odd again
    // ASSERT THAT sieve IS pre-sieved till lps_prime
    //
    // check WORD alignment of min_num, max_num and min_num+N*n_slice:
    ASSERT( 0==(min_num>>1)%WORD_BITS );           // min_num needs to be "left" aligned
    ASSERT( WORD_BITS-1==(max_num>>1)%WORD_BITS ); // max_num needs to be "right" aligned
    ASSERT( 0==n_slice%WORD_BITS ); // IFF (this is OK AND min_num is OK) THEN min_num+N*n_slice WILL BE OK
    double sqrt_max_num = sqrt (max_num);
    uint32_t isqrt_max_num = (uint32_t)(int32_t)sqrt_max_num;
    ASSERT( SQUARE_64(isqrt_max_num) <= max_num && max_num < SQUARE_64(isqrt_max_num+1) );
    ASSERT( double(int32_t(isqrt_max_num)+int32_t(isqrt_max_num)) > double(isqrt_max_num) );
    // cache primes in vector to avoid access to low part of sieve for finding primes to sieve
    std::vector<uint32_t> sieve_prime;        
    uint32_t start_prime = next_prime (sieve, lps_prime);
    sieve_prime.reserve( guess_prim_num_count (start_prime, isqrt_max_num));
    sieve_prime.push_back (start_prime);
    uint64_t prim_num_count=0;
    bool parallel_OK = false;
    uint64_t next_from;
    // so ... let's start!
    for (uint64_t from = min_num; from <= max_num; from += n_slice) {
        uint64_t to = std::min(from+n_slice-2,max_num); 
        prim_num_count += eratosthenes_odd_single_block_seq (sieve, from, to, sieve_prime, parallel_OK, isqrt_max_num);
        if (parallel_OK) {
            next_from = from + n_slice;
            puts ("going parallel");
            goto Parallel;
        }
    }    
    return prim_num_count;
Parallel:
    if (VERBOSE) printf ("num procs %d\n", omp_get_num_procs ());
    omp_set_num_threads (omp_get_num_procs ());
#pragma omp parallel for schedule(dynamic) reduction(+:prim_num_count) 
    for (uint64_t from = next_from; from <= max_num; from += n_slice) {
        uint64_t to = std::min(from+n_slice-2,max_num);
        if (VERBOSE) {
            printf ( "thread %d: from %llu to %llu\n", omp_get_thread_num (), (ULL) from, (ULL) to );
            fflush (stdout);
        }
        prim_num_count += eratosthenes_odd_single_block_par (sieve, from, to, sieve_prime);
    }    
    return prim_num_count;
}


uint32_t adjust_large(uint64_t& large)
// adjust large
// returns: sieve_size
{
    uint64_t old_large = large;
    // We're computing a sieve of odds, so the upper inclusive bound will alwas be odd
    if (!(large&1)) --large;
    uint32_t sieve_size = 1uLL+(large>>1uLL)/uint64_t(WORD_BITS);
    printf("sieve_size=%u sieve-words à 64-bit (8 Bytes)\n",sieve_size);
    // adjust large to what realy fits:
    large = largest_number_in_block (sieve_size);
    printf ("adjusted upper bound from\t%llu\n\t\tto\t\t%llu\t\t(0x%llX hex)\n",
            (ULL)old_large, (ULL)large, (ULL)large);
    ASSERT(large>=old_large-1);
    return sieve_size;
}


uint64_t* compute_huge_prime_numbers (uint64_t large, uint64_t* p_prime_number_count, uint64_t sieve_size)
{
    uint64_t largest_number;
    uint64_t* result = create_huge_memory_block_sieved_with_small_primes (sieve_size, largest_number);
    if (!result) return result;
    ASSERT(largest_number==large);
    enum { LARGEST_PRE_SIEVED_PRIME=17 }; // see above
    uint64_t slice_number = 1+largest_number_in_block ( (L2_CACHE/2)/sizeof(uint64_t) );
    uint64_t prime_number_count = eratosthenes_odd_blockwise (result, 1, large, LARGEST_PRE_SIEVED_PRIME, slice_number);
    if (p_prime_number_count) *p_prime_number_count = prime_number_count;
    return result;
}


#ifndef PRIME_NUMBER_SIEVE_PATH
#define PRIME_NUMBER_SIEVE_PATH "prime-number-sieve"
#endif


enum class Endianes : int8_t
{
    _UNKNOWN_ENDIANNESS=0,
    _BIG_ENDIAN =1,
    _LITTLE_ENDIAN =2,
    _SICK_ENDIAN = 127-3
};


enum class SieveType : int8_t
{
    UNKNOWN_SIEVE_TYPE =0,
    BITWISE_SIEVE_OF_ODDS_ONLY =1,
    BITWISE_ALLNUMS_SIEVE_TYPE =2,
    ONE_NUMBER_PER_BYTE_SIEVE_OF_ODDS_ONLY =3,
    ONE_NUMBER_PER_BYTE_ALLNUMS_SIEVE_TYPE =4
};


#pragma pack(push, 1)
struct sieve_file_header
{
    char sieve_magic[5];
    char szVersion[2];
    char _null[1];
    enum Endianes   svEndian;
    enum SieveType  svType;
    short header_size;
    char _reserved[4];
    uint64_t firstNumberInSieve;
    uint64_t largestNumberInSieve;
};
#pragma pack(pop) // THIS IS OF UTTERMOST IMPORTANCE FOR P.G.O.


enum Endianes detect_host_endianess ()
{

    const uint32_t  endian_test_value=0x01020304;
    const char*     p = reinterpret_cast<const char*>(&endian_test_value);
    if (p[0]<p[1]&&p[1]<p[2]&&p[2]<p[3])
        return Endianes::_BIG_ENDIAN;
    if (p[0]>p[1]&&p[1]>p[2]&&p[2]>p[3])
        return Endianes::_LITTLE_ENDIAN; //e.g. x86
    return Endianes::_SICK_ENDIAN;
}


void write_header (FILE* f, uint64_t upper_bound)
{
    sieve_file_header sh;
    strcpy (sh.sieve_magic,"SIEVE");
    strcpy (sh.szVersion,"01");
    sh.svEndian = detect_host_endianess ();
    sh.svType = SieveType::BITWISE_SIEVE_OF_ODDS_ONLY;
    sh.header_size = sizeof sh;
    memset (sh._reserved, 0, 4);
    sh.firstNumberInSieve = 1;
    sh.largestNumberInSieve = upper_bound;
    auto rc1 = fwrite (&sh,sizeof sh,1,f);
    if (rc1!=1) die("Writing failed: " PRIME_NUMBER_SIEVE_PATH);
}


class Stopwatch
{
    double display_time;
    timespec started;
    bool running;
public:
    Stopwatch() { start(); }

    void start(bool reset=true){
        if (reset) display_time = 0;
        if (clock_gettime(CLOCK_THREAD_CPUTIME_ID,&started))
            die("Stopwatch: clock_gettime failed");
        running = true;
    }
    void stop(){
        if (running){
            timespec now;
            if (clock_gettime(CLOCK_THREAD_CPUTIME_ID,&now))
                die("Stopwatch: clock_gettime failed");
            running = false;
            display_time += (now.tv_sec-started.tv_sec)
                          + (now.tv_nsec-started.tv_nsec)*1e-9;
        }
    }
    double result(){
        if (running){
            stop();
            start(false);
        }
        return display_time;
    }
};


int main ()
{
    FILE* f = fopen (PRIME_NUMBER_SIEVE_PATH, "w");
    if (!f) die (("Cannot open file for writing: " PRIME_NUMBER_SIEVE_PATH));
    cout << "Enter wanted upper limit number: ";
    uint64_t upper_bound = 9999999999uLL;
    cin >> upper_bound;
    size_t sieve_size = adjust_large (upper_bound);
    write_header (f,upper_bound);
    uint64_t prime_number_count=0;
    Stopwatch sw;
    uint64_t* output = compute_huge_prime_numbers (upper_bound, &prime_number_count, sieve_size);
    sw.stop ();
    if (!output) die ("failed.");
    cout << "range [1.."<< upper_bound << "]\n";
    cout << "guessed: " << guess_prim_num_count (2,upper_bound) << " primes.\n";
    cout << "counted: " << prime_number_count << " prime numbers in sieve.\n";
    for (uint64_t i=upper_bound; i>3; i-=2) {
        if (/* HACK sieve negated during counting!
             * IS_COMPOSITE now does the IS_PRIME test!
             */IS_COMPOSITE(output,i)) {
            cout << "largest prime number in sieve range is " << i << '\n';
            break;
        }
    }
    cout << " (time used: " << sw.result () << " seconds.)\n";
    cout << "saving " PRIME_NUMBER_SIEVE_PATH " .." << endl;
    auto rc=fwrite (output, sizeof(uint64_t), sieve_size, f);
    fclose (f);
    return rc==sieve_size ? EXIT_SUCCESS : EXIT_FAILURE;
}
