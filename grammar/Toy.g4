grammar Toy;

file: stat* EOF;

stat
    : funcStat
    | blockStat
    | declStat END
    | printStat END
    | returnStat END
    ;

funcStat
    : FUNCTION ID LEFT_BRACKET params? RIGHT_BRACKET blockStat
    ;

blockStat
    : LEFT_CURLY stat* RIGHT_CURLY
    ;

declStat
    : VAR ID shape? ASSN expr
    ;

printStat
    : PRINT LEFT_BRACKET expr? RIGHT_BRACKET
    ;

returnStat
    : RETURN expr?
    ;

shape
    : LANGLE INT (COMMA INT)+? RANGLE
    ;

expr
    : LEFT_BRACKET expr RIGHT_BRACKET                   #bracket
    // | DOT_NOTATION                                      #dot
    // | <assoc='right'> expr POWER expr                   #power
    | expr (op=MUL | op=DIV | op=DSTAR) expr            #mulDivMatmul
    | expr (op=ADD | op=SUB) expr                       #addSub
    | funcExpr                                          #call
    | ID                                                #variable
    | slice                                             #literalSlice
    ;

funcExpr
    : ID LEFT_BRACKET (expr ( COMMA expr )*)? RIGHT_BRACKET
    ;

slice
    : LEFT_SQUARE sliceValue (COMMA sliceValue)* RIGHT_SQUARE #literal
    | LEFT_SQUARE (slice (COMMA slice)*)? RIGHT_SQUARE #nested
    ;

sliceValue
    : INT
    | FLOAT
    ;

params : ID (COMMA ID)*;

RETURN : 'return';
PRINT : 'print';
END : ';';
LEFT_BRACKET: '(';
RIGHT_BRACKET: ')';
LEFT_SQUARE: '[';
RIGHT_SQUARE: ']';
LEFT_CURLY: '{';
RIGHT_CURLY: '}';
LANGLE : '<';
RANGLE : '>';
VAR : 'var';
FUNCTION : 'def';
ASSN : '=';
ADD: '+';
SUB: '-';
MUL: '*';
DSTAR: '**';
DIV: '/';
MOD: '%';
GE: '>=';
LE: '<=';
NE: '!=';
EQ: '==';
POWER: '^';
DBAR: '||';
COMMA: ',';
IN_ARROW: '<-';
OUT_ARROW: '->';

// LITERALS
DOT_NOTATION: ID_FRAGMENT DOT (ID_FRAGMENT | DIGIT+);
FLOAT
    : '-'? DIGIT+ DOT DIGIT+ E_NOTATION?
    | '-'? DOT DIGIT+ E_NOTATION?
    | '-'? DIGIT+ DOT E_NOTATION?
    | '-'? DIGIT+ E_NOTATION
    ;
INT: DIGIT+;
ID: ID_FRAGMENT;

// FRAGMENTS
fragment E_NOTATION: [Ee] [+-]? [0-9]+;
fragment DIGIT: [0-9];
fragment ID_FRAGMENT: [a-zA-Z_] [a-zA-Z0-9_]*;
fragment DOT: '.';

// Skip whitespace
WS : [ \t\r\n]+ -> skip ;
COMMENT : '#' .*? '\n' -> skip ;
