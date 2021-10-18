#include <stdio.h>
#include <stdlib.h>
/**
 calc_n()
 input: n
 output: n!
*/
int calc_n(int n){
 if(n==1){
 return 1;
 }
 else{
 return calc_n(n-1)*n;
 }
}
/**
 sort
*/
//exchange the 2 items a and b
void swap(int a, int b){
 int tmp = a;
 a = b;
 b = tmp;
}
void BinaryInsertSort(int *arr, int len){
 for (int i = 1; i < len; ++i){
 int tmp = arr[i];
 int iHigh = i - 1;
 int iLow = 0;
 int iMid = 0;
 while (iLow <= iHigh){
 iMid = (iLow + iHigh) / 2;
 if (tmp > arr[iMid] ){
 iLow = iMid + 1;
 }
 else{
 iHigh = iMid - 1;
 }
 }
 for (int j = i; j > iMid; --j){
 arr[j] = arr[j - 1];
 }
 arr[iLow] = tmp;
 }
}
/*********************************************************************
*
* main()
*
* Function description
* Application entry point.
*/
int main(void) {
 int i;
 //calculate 10!
 int nn = calc_n(3);
 printf("calculate n!: %d \n", nn);
 //test sort
 int buf[10] = { 15, 45, 4, 7, 10, 235, 31, 52, 988, 15 };
 int m = sizeof(buf);
 printf("before sort: \n ");
 for(i = 0; i <sizeof(buf) / sizeof(int); i++){
 printf("%d ", buf[i]);
 }
 BinaryInsertSort(buf, sizeof(buf) / sizeof(int));
 printf("\nafter sort: \n ");
 for(i = 0; i <sizeof(buf) / sizeof(int); i++){
 printf("%d ", buf[i]);
 }
 
}
