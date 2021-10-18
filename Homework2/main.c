#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
 
#define PI 3.1415926
 
#define	height	128
#define	width	128
 
typedef unsigned char  BYTE;	// Define Byte type , using space 1 byte
 
int main()
{
	FILE *fp = NULL;        // Using to read file
	
	BYTE Input[height][width];  // Store image pixel

	// Raw image path: input and output
	char path[256] = "/Users/admin/Downloads/butter128.raw";
	char outpath[256] = "/Users/admin/Downloads/butter128out.raw";
	
	int i,j;
	
	// Read image
	printf("Start reading raw image!\n");
	if((fp = fopen( path, "rb" )) == NULL)
	{
	    printf("Can not open the raw image!\n" );
	    return 0;
	}
	else
        {
            printf("Read image OK!\n");
        } 
	
	for( i = 0; i < height; i++ )
	{
            for( j = 0; j < width ; j ++ )
		{
                fread( &Input[i][j], 1, 1, fp );
    		//printf("%d  \t",Input[i][j]);       //print all piexl
		}
	}
        printf("Read all pixel!\n");
	fclose(fp);

	// rotate image
        double angle = -45;   //giving a anger +：逆时针

        double sita = angle*PI / 180;

        // get the new size of output image
        double a = (width - 1) / 2.0;
	double b = (height - 1) / 2.0;
 
	double x1 = -a*cos(sita) - b*sin(sita);
	double y1 = -a*sin(sita) + b*cos(sita);
 
	double x2 = a*cos(sita) - b*sin(sita);
	double y2 = a*sin(sita) + b*cos(sita);
 
	double x3 = a*cos(sita) + b*sin(sita);
	double y3 = a*sin(sita) - b*cos(sita);
 
	double x4 = -a*cos(sita) + b*sin(sita);
	double y4 = -a*sin(sita) - b*cos(sita);
 
	int wo = round(fmax(abs(x1 - x3), abs(x2 - x4)));
	int ho = round(fmax(abs(y1 - y3), abs(y2 - y4)));

        double centerX = (width+1)/2.0;
        double centerY = (height+1)/2.0;

        // init convert parameters
        double alph = cos(sita);
        double beta = sin(sita);

        double M11 = alph;
        double M12 = beta;
        double M13 = (1-alph)*centerX-beta*centerY;
        double M21 = -beta;
        double M22 = alph;
        double M23 = (1-alph)*centerY+beta*centerX;
       
        BYTE Output[height][width];
        //BYTE Output[ho][wo];

        // Write image
        printf("Start rotating and writing raw image!\n");
	if( ( fp = fopen( outpath, "wb" ) ) == NULL )
	{
	    printf("Can not create the raw_image : %s\n", outpath );
	    return 0;
	}

        for( i = 0; i < height; i++ )
	{
	    for( j = 0; j < width; j ++ )
		{
                // rotate image
                int ox = round(M11*i+M12*j+M13);
                int oy = round(M21*i+M22*j+M23);

                // solve the egde
                if(ox < 0 || ox > width-1 || oy < 0 || oy > height-1){
                    Output[i][j] = 0;
                }
                else{
                    Output[i][j] = Input[ox][oy];
                }
                // write to raw image
		fwrite( &Output[i][j], 1 , 1, fp );
		}
	}

        printf("Write image OK!\n");

	fclose(fp);

        return 0;
}