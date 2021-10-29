	AREA     ARMex, CODE, READONLY                                	
    ENTRY
	
	CODE32
	
	; MOV Opcode
    MOV	r1, #5
	MOV	r2, #2
	
	; 32bit ARM
	UMULL r3, r4, r1, r2
	
	ADR r0, mul16 + 1
	
	BX r0
	
	CODE16	
mul16	

	; MOV Opcode
    MOV	r4, #5
	MOV	r5, #2
	
	; 16bit Thumb
	MUL r5, r4

	END 
