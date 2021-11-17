		AREA 	TopSwi, CODE, READONLY
		IMPORT	C_SWI_Handler 
		EXPORT     	SWI_Handler
			
SWI_Handler
    	STMFD      	sp!,{r0-r12,lr}
    	LDR        	r0, [lr,#-4] 		
		BIC        	r0, r0,#0xff000000 	
		MOV	    	R1, SP   		
     	BL 			C_SWI_Handler 		
   		;LDMFD       sp!, {r0-r12,pc}^
	   	END
