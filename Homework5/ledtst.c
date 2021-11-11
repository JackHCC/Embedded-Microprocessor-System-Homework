
#include<stdlib.h>
#include<stdio.h>
int *rGPFCON = (int *) 0x56000050;
int *rGPFDAT = (int *) 0x56000054;
int delay(int times);

int main(void)
{
	*rGPFCON=0x5500;
	while(1)
	{
		*rGPFDAT = 0x00;
		delay(500); 
		*rGPFDAT = 0xf0;
		delay(500);
	}
	return 1;
}

int delay(int times)
{
	int i,j;
	for(i=0;i<times;i++)
		for(j=0;j<times;j++)
		{
		
		}
		return 1;
}

