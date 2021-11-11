
                

MC_BASE         EQU     0x48000000      ; Memory Controller Base Address
BWSCON_OFS      EQU     0x00            ; Bus Width and Wait Status Ctrl Offset
BANKCON0_OFS    EQU     0x04            ; Bank 0 Control Register        Offset
BANKCON1_OFS    EQU     0x08            ; Bank 1 Control Register        Offset
BANKCON2_OFS    EQU     0x0C            ; Bank 2 Control Register        Offset
BANKCON3_OFS    EQU     0x10            ; Bank 3 Control Register        Offset
BANKCON4_OFS    EQU     0x14            ; Bank 4 Control Register        Offset
BANKCON5_OFS    EQU     0x18            ; Bank 5 Control Register        Offset
BANKCON6_OFS    EQU     0x1C            ; Bank 6 Control Register        Offset
BANKCON7_OFS    EQU     0x20            ; Bank 7 Control Register        Offset
REFRESH_OFS     EQU     0x24            ; SDRAM Refresh Control Register Offset
BANKSIZE_OFS    EQU     0x28            ; Flexible Bank Size Register    Offset
MRSRB6_OFS      EQU     0x2C            ; Bank 6 Mode Register           Offset
MRSRB7_OFS      EQU     0x30            ; Bank 7 Mode Register           Offset				
				
				
MC_SETUP        EQU     0
BWSCON_Val      EQU     0x22111110
BANKCON0_Val    EQU     0x00000700
BANKCON1_Val    EQU     0x00000700
BANKCON2_Val    EQU     0x00000700
BANKCON3_Val    EQU     0x00000700
BANKCON4_Val    EQU     0x00000700
BANKCON5_Val    EQU     0x00000700
BANKCON6_Val    EQU     0x00018005
BANKCON7_Val    EQU     0x00018005
REFRESH_Val     EQU     0x008e0459
BANKSIZE_Val    EQU     0x000000b2
MRSRB6_Val      EQU     0x00000030
MRSRB7_Val      EQU     0x00000030

				AREA  SDRAMINI, CODE, readonly
				EXPORT INISDRAM
				

INISDRAM
				LDR     R0, =MC_BASE
                LDR     R1,      =BWSCON_Val
                STR     R1, [R0, #BWSCON_OFS]
                LDR     R1,      =BANKCON0_Val
                STR     R1, [R0, #BANKCON0_OFS]
                LDR     R1,      =BANKCON1_Val
                STR     R1, [R0, #BANKCON1_OFS]
                LDR     R1,      =BANKCON2_Val
                STR     R1, [R0, #BANKCON2_OFS]
                LDR     R1,      =BANKCON3_Val
                STR     R1, [R0, #BANKCON3_OFS]
                LDR     R1,      =BANKCON4_Val
                STR     R1, [R0, #BANKCON4_OFS]
                LDR     R1,      =BANKCON5_Val
                STR     R1, [R0, #BANKCON5_OFS]
                LDR     R1,      =BANKCON6_Val
                STR     R1, [R0, #BANKCON6_OFS]
                LDR     R1,      =BANKCON7_Val
                STR     R1, [R0, #BANKCON7_OFS]
                LDR     R1,      =REFRESH_Val
                STR     R1, [R0, #REFRESH_OFS]
                MOV     R1,      #BANKSIZE_Val
                STR     R1, [R0, #BANKSIZE_OFS]
                MOV     R1,      #MRSRB6_Val
                STR     R1, [R0, #MRSRB6_OFS]
                MOV     R1,      #MRSRB7_Val
                STR     R1, [R0, #MRSRB7_OFS]
				
				MOV		pc,lr
                END          	              
