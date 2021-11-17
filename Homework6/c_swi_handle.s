 		AREA SecondSwi, CODE, READONLY 
		EXPORT     C_SWI_Handler
C_SWI_Handler 
		CMP    r0, #0x01000000          ; Range check?
    	LDRLE  pc, [pc,r0,LSL #2] 		;(PC->DCD SWInum0)
		B      	SWIOutOfRange 
SWIJumpTable	
		DCD    SWInum0
    	DCD    SWInum1
SWInum0   ; SWI number 0 code
    	B    EndofSWI
SWInum1   ; SWI number 1 code
    	B    EndofSW

EndofSWI		
		LDMFD   sp!, {r0-r3}	
		
		ADDS	r0 , r0, r2
		ADC		r1 , r1, r3	
		
		SUB 	lr, lr, #4
		LDMFD   sp!, {r4-r12,pc}^

EndofSW	
		LDMFD   sp!, {r0-r1}
		SMULL	r2, r3, r0, r1
		MOV		r0, r2
		MOV		r1, r3
		
		
		SUB 	lr, lr, #4
		LDMFD   sp!, {r2-r12,pc}^
	   	

SWIOutOfRange 
		SUB 	lr, lr, #4
		LDMFD   sp!, {r0-r12,pc}^

		END
		