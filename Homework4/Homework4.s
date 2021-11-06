	; implement LDRUA
	MACRO 	
	LDRUA	$RD,$RS
	LDRB	v2,	[$RS]
	LDRB	v3,	[$RS, #1]
	LDRB	v4,	[$RS, #2]
	LDRB	v5,	[$RS, #3]

	ADD		v6,	v2,	v3,	LSL	#8
	ADD		v6,	v6,	v4,	LSL	#16
	ADD		v6,	v6,	v5,	LSL	#24

	MOV		$RD,v6
	MEND
	
	; implement STRUA
	MACRO
	STRUA	$RD,$RS
	MOV		v2,	$RD
	STRB	v2,	[$RS]
	LSR		v2,	#8
	STRB	v2,	[$RS,#1]	
	LSR		v2,	#8
	STRB	v2,	[$RS,#2]
	LSR		v2,	#8
	STRB	v2,	[$RS,#3]
	MEND

	AREA     ARMex, CODE, READONLY                                	
    ENTRY
	CODE32
	
	; Test LDRUA
	LDR		R1,	=P1
	LDR		R2,	=P2
	LDR		R3,	=P3
	LDR		R4,	=P4
	
	LDRUA	R1,	R1
	LDRUA	R2,	R2
	LDRUA	R3,	R3
	LDRUA	R4,	R4
	
	
	; Test STRUA
	MOV		R1, 0xFFFFFFFF
	LDR		R0,	=P2
	STRUA	R1,	R0
	
	
P1	DCB		0x99
P2	DCDU	0x77777777
P3	DCWU	0x6655
P4	DCDU	0x55555555

	END 
