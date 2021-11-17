#include <stdio.h>
#include "swi.h"
unsigned *swi_vec = (unsigned *)0x08;
extern void SWI_Handler(void);

unsigned Install_Handler (unsigned *handlerloc, unsigned *vector)
{  
    unsigned vec, oldvec;
    vec = ((unsigned)handlerloc - (unsigned)vector - 0x8)>>2;
    if ((vec & 0xFF000000) != 0)
   	 { return 0;}
    vec = 0xEa000000 | vec;
    oldvec = *vector;
    *vector = vec;
    return (oldvec);
}

int main( void )
{	 
    long long res; 
	
		long long res1;
	
		long long a = 320000;
		long long b = 640000;
	
		int al = a;
		int ah = a >> 32;
		int bl = b;
		int bh = b >> 32;
    
    Install_Handler((unsigned *) SWI_Handler, swi_vec);
	
    res = add_two(al, ah, bl, bh);
		res1 = mut_two(al, bl);

    return 0;
}

