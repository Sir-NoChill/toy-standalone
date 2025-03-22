grammar Toy;

file: .*? EOF;


// Skip whitespace
WS : [ \t\r\n]+ -> skip ;
COMMENT : '#' .*? '\n' -> skip ;
