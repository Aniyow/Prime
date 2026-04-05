#include <bits/stdc++.h>
using namespace std;

int main() {
    long long n = 2000000;

    vector<bool> isPrime(n+1, true);
    isPrime[0] = isPrime[1] = false;

    for (long long i = 2; i * i <= n; i++) {
        if (isPrime[i]) {
            for (long long j = i * i; j <= n; j += i) {
                isPrime[j] = false;
            }
        }
    }

    ofstream file("primes_with_gaps.csv");
    file << "prime,gap\n";

    long long prev = -1;

    for (long long i = 2; i <= n; i++) {
        if (isPrime[i]) {
            if (prev == -1) {
                file << i << ",0\n";  
            } else {
                file << i << "," << (i - prev) << "\n";
            }
            prev = i;
        }
    }

    file.close();

    cout << "Saved primes_with_gaps.csv\n";
    return 0;
}