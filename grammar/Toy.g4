grammar Toy;

file: stat* EOF;

stat
    : funcStat
    ;

scopeStat
    : blockStat
    | declStat END
    | RETURN expr? END
    ;

funcStat
    : FUNCTION ID LEFT_BRACKET params? RIGHT_BRACKET blockStat
    ;

blockStat
    : LEFT_CURLY scopeStat* RIGHT_CURLY
    ;

declStat
    : VAR ID shape? ASSN expr
    ;

shape
    : LANGLE INT (COMMA INT)*? RANGLE
    ;

expr
    : LEFT_BRACKET expr RIGHT_BRACKET                   #bracket
    // | DOT_NOTATION                                      #dot
    | <assoc='right'> expr POWER expr                   #power
    | expr (op=MUL | op=DIV | op=DSTAR) expr            #mulDivMod
    | expr (op=ADD | op=SUB) expr                       #addSub
    | funcExpr                                          #func
    | INT 						#literal
    | FLOAT						#literal
    | ID                                                #variable
    | slice                                             #literalSlice
    ;

funcExpr
    : ID LEFT_BRACKET RIGHT_BRACKET
    | ID LEFT_BRACKET expr ( COMMA expr )* RIGHT_BRACKET
    ;

slice
    : LEFT_SQUARE (expr (COMMA expr)*)? RIGHT_SQUARE
    ;
params : ID (COMMA ID)*;


RETURN : 'return';
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

NUMBER : INT | FLOAT ;

// FRAGMENTS
fragment E_NOTATION: [Ee] [+-]? [0-9]+;
fragment DIGIT: [0-9];
fragment ID_FRAGMENT: [a-zA-Z_] [a-zA-Z0-9_]*;
fragment DOT: '.';

// Skip whitespace
WS : [ \t\r\n]+ -> skip ;
COMMENT : '#' .*? '\n' -> skip ;
