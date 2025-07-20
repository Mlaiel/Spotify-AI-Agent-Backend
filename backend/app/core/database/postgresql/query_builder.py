"""
PostgreSQL Query Builder Module - Professional Implementation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from psycopg2 import sql

logger = logging.getLogger(__name__)

class QueryBuilder:
    """Professional PostgreSQL Query Builder"""
    
    def __init__(self, table_name: str):
        """Initialize QueryBuilder for specific table"""
        if not table_name or not isinstance(table_name, str):
            raise ValueError("Table name must be a non-empty string")
        self.table_name = table_name.strip()
        logger.debug(f"QueryBuilder initialized for table: {self.table_name}")
    
    def select(self, columns=None, where_conditions=None, limit=None):
        """Build SELECT query"""
        fields_sql = sql.SQL('*') if columns is None else sql.SQL(', ').join(map(sql.Identifier, columns))
        query = sql.SQL("SELECT {fields} FROM {table}").format(
            fields=fields_sql, table=sql.Identifier(self.table_name)
        )
        parameters = []
        if where_conditions:
            where_parts = [sql.SQL("{} = %s").format(sql.Identifier(col)) for col in where_conditions.keys()]
            query = query + sql.SQL(" WHERE ") + sql.SQL(' AND ').join(where_parts)
            parameters = list(where_conditions.values())
        if limit:
            query = query + sql.SQL(" LIMIT %s")
            parameters.append(limit)
        return query, parameters
    
    def insert(self, data):
        """Build INSERT query"""
        columns = list(data.keys())
        values = list(data.values())
        query = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING id").format(
            table=sql.Identifier(self.table_name),
            columns=sql.SQL(', ').join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(', ').join([sql.Placeholder()] * len(columns))
        )
        return query, values
    
    def update(self, data, where_conditions):
        """Build UPDATE query"""
        set_columns = list(data.keys())
        set_values = list(data.values())
        where_columns = list(where_conditions.keys())
        where_values = list(where_conditions.values())
        
        set_clause = sql.SQL(', ').join([sql.SQL("{} = %s").format(sql.Identifier(col)) for col in set_columns])
        where_clause = sql.SQL(' AND ').join([sql.SQL("{} = %s").format(sql.Identifier(col)) for col in where_columns])
        
        query = sql.SQL("UPDATE {table} SET {set_clause} WHERE {where_clause}").format(
            table=sql.Identifier(self.table_name), set_clause=set_clause, where_clause=where_clause
        )
        return query, set_values + where_values
    
    def delete(self, where_conditions):
        """Build DELETE query"""
        columns = list(where_conditions.keys())
        values = list(where_conditions.values())
        where_clause = sql.SQL(' AND ').join([sql.SQL("{} = %s").format(sql.Identifier(col)) for col in columns])
        query = sql.SQL("DELETE FROM {table} WHERE {where_clause}").format(
            table=sql.Identifier(self.table_name), where_clause=where_clause
        )
        return query, values

__all__ = ["QueryBuilder"]
